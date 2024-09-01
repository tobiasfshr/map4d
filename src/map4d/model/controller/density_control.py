from typing import Optional

import torch
from gsplat._torch_impl import quat_to_rotmat
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.comms import get_world_size
from torch import Tensor
from torch import distributed as dist

from map4d.model.struct.gaussians import Gaussians


class VanillaAdaptiveDensityController:
    """Vanilla adaptive density controller used in 3DGS."""

    def __init__(
        self,
        gaussians: Gaussians,
        update_gauss_fn: callable,
        warmup_length: int,
        refine_every: int,
        reset_alpha_every: int,
        stop_split_at: int,
        continue_cull_post_densification: bool,
        num_train_data: int,
        cull_alpha_thresh: float,
        cull_scale_thresh: float,
        densify_grad_thresh: float,
        densify_size_thresh: float,
        n_split_samples: int,
        cull_screen_size: float,
        split_screen_size: float,
        stop_screen_size_at: int,
        scene_box: SceneBox | None = None,
        min_gaussians: int = 1024,
        max_gaussians: int = 0,
        size_frac: float = 1.6,
        min_num_visible: int | float = 0,
    ) -> None:
        super().__init__()
        self.gaussians = gaussians
        self.scene_box = scene_box
        self._update_gauss_fn = update_gauss_fn

        # schedule parameters
        self.warmup_length = warmup_length
        self.refine_every = refine_every
        self.reset_alpha_every = reset_alpha_every
        self.stop_split_at = stop_split_at
        self.stop_screen_size_at = stop_screen_size_at
        self.continue_cull_post_densification = continue_cull_post_densification
        self.num_train_data = num_train_data

        # control parameters
        self.cull_alpha_thresh = cull_alpha_thresh
        self.cull_scale_thresh = cull_scale_thresh
        self.densify_grad_thresh = densify_grad_thresh
        self.densify_size_thresh = densify_size_thresh
        self.n_split_samples = n_split_samples
        self.size_frac = size_frac
        self.cull_screen_size = cull_screen_size
        self.split_screen_size = split_screen_size
        self.min_gaussians = min_gaussians
        self.max_gaussians = max_gaussians
        if isinstance(min_num_visible, float):
            min_num_visible = int(min_num_visible * refine_every)
        self.min_num_visible = min_num_visible

        self.world_size = get_world_size()
        if min_num_visible > 0:
            assert self.world_size == 1, "min_num_visible is not supported in distributed training."

        self.xys_grad_norm = None
        self.vis_counts = None
        self.max_2Dsize = None
        self.last_size = None
        self.num_total_gaussians = None

    @property
    def num_gaussians(self) -> int:
        """Return number of Gaussians in the Gaussian field that is controlled by this controller."""
        return self.gaussians.num_points

    def set_num_total_gaussians(self, num_gaussians: int) -> None:
        """Set the number of Gaussians ."""
        self.num_total_gaussians = num_gaussians

    def _culling(
        self,
        means: Tensor,
        opacities: Tensor,
        scales: Tensor,
        extra_cull_mask: Optional[torch.Tensor] = None,
        cull_huge_gaussians: bool = False,
        cull_huge_screen_size: bool = False,
    ) -> torch.Tensor:
        """This function deletes gaussians with under a certain opacity threshold.

        Args:
            means: a list of means
            opacities: a list of opacities
            scales: a list of scales
            extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
            cull_huge_gaussians: whether to cull huge gaussians
            cull_huge_screen_size: whether to cull huge screen size gaussians (only when also culling huge gaussians)
        """
        culls = (opacities < self.cull_alpha_thresh).squeeze()

        if cull_huge_gaussians:
            toobigs = (scales.max(dim=-1).values > self.cull_scale_thresh).squeeze()
            # cull huge ones (within the scene box)
            if self.scene_box is not None:
                within_bounds = self.scene_box.within(means)
                toobigs = toobigs & within_bounds
            if cull_huge_screen_size:
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.cull_screen_size).squeeze()
            culls = culls | toobigs

        if (~culls).sum().item() < self.min_gaussians:
            # if we have too few points, don't cull
            culls = torch.zeros_like(culls)

        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask

        return culls

    def _splitting(self, scales: Tensor, quats: Tensor, mask: Tensor) -> Gaussians:
        """This function splits gaussians that are too large."""
        n_splits = mask.sum().item()
        if n_splits == 0:
            return Gaussians.empty_like(self.gaussians)

        samps = self.n_split_samples
        centered_samples = torch.randn(
            (samps * n_splits, 3), device=self.gaussians.device
        )  # Nx3 of axis-aligned scales
        # broadcast centered samples from rank 0 in DDP
        if self.world_size > 1:
            dist.broadcast(centered_samples, 0)

        scaled_samples = scales[mask].repeat(samps, 1) * centered_samples
        quats = quats[mask] / quats[mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()

        new_means = rotated_samples + self.gaussians.means[mask].repeat(samps, 1)
        new_opacities = self.gaussians.opacities[mask].repeat(samps, 1)
        new_scales = torch.log(torch.exp(self.gaussians.scales[mask]) / self.size_frac).repeat(samps, 1)
        new_quats = self.gaussians.quats[mask].repeat(samps, 1)
        new_other_attrs = (
            {key: getattr(self.gaussians, key)[mask].repeat(samps, 1) for key in self.gaussians._other_keys}
            if hasattr(self.gaussians, "_other_keys")
            else None
        )
        new_gaussians = Gaussians(
            means=new_means,
            opacities=new_opacities,
            scales=new_scales,
            quats=new_quats,
            other_attrs=new_other_attrs,
        )
        return new_gaussians

    @torch.no_grad()
    def update_statistics(self, step: int, visible_mask: Tensor, xy_grads: Tensor, radii: Tensor):
        # to save some training time, we no longer need to update those stats post refinement
        if step >= self.stop_split_at:
            return
        # keep track of a moving average of grad norms
        if self.xys_grad_norm is None:
            self.vis_counts = torch.zeros(self.num_gaussians, device=xy_grads.device, dtype=torch.float32)
            self.xys_grad_norm = torch.zeros_like(self.vis_counts)
        else:
            assert self.vis_counts is not None

        self.vis_counts[visible_mask] += 1
        self.xys_grad_norm[visible_mask] += xy_grads

        # NOTE: We do only optionally split / cull based on screen size as it was not activated in 3DGS
        # see https://github.com/graphdeco-inria/gaussian-splatting/issues/544
        if self.split_screen_size > 0:
            assert self.last_size is not None
            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_gaussians, device=radii.device, dtype=torch.float32)
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                radii / float(max(self.last_size[0], self.last_size[1])),
            )

    def sync_statistics(self):
        """Sync statistics between different GPUs if they are tracked still."""
        if self.vis_counts is None and self.xys_grad_norm is None and self.max_2Dsize is None:
            return
        if self.world_size > 1:
            dist.barrier()
            if self.vis_counts is not None:
                dist.all_reduce(self.vis_counts, op=dist.ReduceOp.SUM)
            if self.xys_grad_norm is not None:
                dist.all_reduce(self.xys_grad_norm, op=dist.ReduceOp.SUM)
            if self.max_2Dsize is not None:
                dist.all_reduce(self.max_2Dsize, op=dist.ReduceOp.MAX)

    def _update_params_in_optimizers(
        self, num_splits: int, num_dups: int, culls: Tensor, index: int, optimizers: Optimizers
    ):
        # NOTE: this code assumes every partial Gaussian field has the same param groups
        # remove, duplicate in optimizer. First remove from existing, then add new (zeros)
        for group, param in self.gaussians.get_param_groups().items():
            remove_from_optim(optimizers.optimizers[group], index, culls, param)
        for group, param in self.gaussians.get_param_groups().items():
            dup_in_optim(optimizers.optimizers[group], index, [num_splits, num_dups], param)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, step: int, param_index: int, optimizers: Optimizers) -> None:
        if step <= self.warmup_length:
            return

        # if we haven't had enough observations, don't do anything
        if self.vis_counts is not None and self.min_num_visible > 0:
            if not (self.vis_counts > self.min_num_visible).any():
                return 0, 0, 0

        reset_interval = self.reset_alpha_every * self.refine_every
        do_densification = step < self.stop_split_at and step % reset_interval > min(
            reset_interval // 2, self.num_train_data + self.refine_every
        )
        do_culling = step >= self.stop_split_at and self.continue_cull_post_densification
        cull_huge_gaussians = step > self.refine_every * self.reset_alpha_every
        cull_huge_screen_size = step < self.stop_screen_size_at

        if not (do_densification or do_culling):
            return 0, 0, 0

        self.sync_statistics()
        means, opacities, scales, quats = (
            self.gaussians.means,
            self.gaussians.opacities,
            self.gaussians.scales,
            self.gaussians.quats,
        )
        opacities = torch.sigmoid(opacities)
        scales = torch.exp(scales)

        if do_densification:
            assert self.xys_grad_norm is not None and self.vis_counts is not None
            avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])

            # which gaussians received high gradients?
            high_grads = (avg_grad_norm > self.densify_grad_thresh).squeeze()

            # which gaussians are too big? (within bounds)
            splits = (scales.max(dim=-1).values > self.densify_size_thresh).squeeze()
            if self.scene_box is not None:
                within_bounds = self.scene_box.within(means)
                splits &= within_bounds
            splits &= high_grads

            # which gaussians are too big on screen?
            if step < self.stop_screen_size_at:
                toobigs = (self.max_2Dsize > self.split_screen_size).squeeze()
                splits |= toobigs

            # which gaussians are too small?
            dups = (scales.max(dim=-1).values <= self.densify_size_thresh).squeeze()
            dups &= high_grads

            # cap the growing to max_gaussians
            if self.max_gaussians > 0 and self.num_total_gaussians + (dups | splits).sum() > self.max_gaussians:
                splits = torch.zeros_like(splits)
                dups = torch.zeros_like(dups)

            split_gaussians = self._splitting(scales, quats, splits)
            dup_gaussians = self.gaussians[dups]
        else:
            splits = None
            split_gaussians = Gaussians.empty_like(self.gaussians)
            dup_gaussians = Gaussians.empty_like(self.gaussians)

        culls = self._culling(means, opacities, scales, splits, cull_huge_gaussians, cull_huge_screen_size)

        # modify gaussians in-place / params in optimizers
        updated_gaussians = self.gaussians[~culls].detach().cat(split_gaussians.cat(dup_gaussians))
        self.gaussians = self._update_gauss_fn(updated_gaussians, param_index)
        num_dups, num_splits, num_culls = dup_gaussians.num_points, split_gaussians.num_points, culls.sum().item()
        self._update_params_in_optimizers(num_splits, num_dups, culls, param_index, optimizers)

        if (
            step < self.stop_split_at
            and step % reset_interval == self.refine_every
            and self.gaussians.opacities is not None
        ):
            # Reset value is set to be twice of the cull_alpha_thresh
            reset_value = self.cull_alpha_thresh * 2.0
            self.gaussians.opacities.data = torch.clamp(
                self.gaussians.opacities.data, max=torch.logit(torch.tensor(reset_value, device=self.device)).item()
            )
            # reset the exp of optimizer
            optim = optimizers.optimizers["opacity"]
            param = optim.param_groups[0]["params"][param_index]
            param_state = optim.state[param]
            param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
            param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

        # reset statistics, return stats
        self.xys_grad_norm = None
        self.vis_counts = None
        self.max_2Dsize = None
        return num_dups, num_splits, num_culls


def remove_from_optim(optimizer, i, deleted_mask, new_param):
    """removes the deleted_mask from the optimizer provided"""
    param = optimizer.param_groups[0]["params"][i]
    param_state = optimizer.state[param]
    del optimizer.state[param]

    # Modify the state directly without deleting and reassigning.
    if "exp_avg" in param_state.keys():
        param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
        param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

    # Update the parameter in the optimizer's param group.
    optimizer.state[new_param] = param_state
    optimizer.param_groups[0]["params"][i] = new_param


def dup_in_optim(optimizer, i, total_dups, new_param):
    """adds the parameters to the optimizer"""
    param = optimizer.param_groups[0]["params"][i]

    param_state = optimizer.state[param]
    for total_dup in total_dups:
        if "exp_avg" in param_state.keys():
            dtype, device = param_state["exp_avg"].dtype, param_state["exp_avg"].device
            zeros = torch.zeros((total_dup, *param_state["exp_avg"].shape[1:]), dtype=dtype, device=device)
            param_state["exp_avg"] = torch.cat([param_state["exp_avg"], zeros], dim=0)
            param_state["exp_avg_sq"] = torch.cat([param_state["exp_avg_sq"], zeros], dim=0)

    # Update the parameter in the optimizer's param group.
    del optimizer.state[param]
    optimizer.state[new_param] = param_state
    optimizer.param_groups[0]["params"][i] = new_param
    del param
