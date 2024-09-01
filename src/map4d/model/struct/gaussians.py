from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from torch.nn import Parameter


class Gaussians(torch.nn.Module):
    """Data structure for 3D Gaussian representation."""

    def __init__(
        self,
        means: Tensor,
        opacities: Tensor,
        scales: Tensor,
        quats: Tensor,
        other_attrs: dict | None = None,
    ) -> None:
        super().__init__()
        self.means = Parameter(means)

        assert len(opacities.shape) == 2, "Expected shape N, 1"
        self._opacities = Parameter(opacities)
        assert len(self.means) == len(self._opacities)

        self._scales = Parameter(scales)
        assert len(self.means) == len(self._scales)

        self._quats = Parameter(quats)
        assert len(self.means) == len(self._quats)

        self._other_keys = []
        if other_attrs is not None:
            self._other_keys = list(other_attrs.keys())
            for key, value in other_attrs.items():
                assert len(value.shape) == 2, "Expected shape N, C"
                setattr(self, key, Parameter(value, requires_grad=value.requires_grad))
                assert len(self.means) == len(value)

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def quats(self):
        if hasattr(self, "_quats"):
            return self._quats
        return None

    @property
    def scales(self):
        if hasattr(self, "_scales"):
            return self._scales
        return None

    @property
    def opacities(self):
        if hasattr(self, "_opacities"):
            return self._opacities
        return None

    @property
    def others(self):
        if hasattr(self, "_other_keys"):
            return {key: getattr(self, key) for key in self._other_keys}
        return None

    @property
    def device(self):
        return self.means.device

    def get_param_groups(self) -> Dict[str, Parameter]:
        """Get Gaussian parameters for optimization."""
        params = {"xyz": self.means}
        params["opacity"] = self.opacities
        params["scales"] = self.scales
        params["quats"] = self.quats
        for k in self._other_keys:
            v = getattr(self, k)
            if v.requires_grad:
                params[k] = v
        return params

    def add_attribute(self, key: str, value: Tensor, requires_grad: bool = True):
        """Add an attribute to the Gaussians object."""
        assert len(self.means) == len(value)
        setattr(self, key, Parameter(value, requires_grad=requires_grad))
        if not hasattr(self, "_other_keys"):
            self._other_keys = []
        self._other_keys.append(key)

    def cat(self, other: Gaussians):
        means = torch.cat([self.means.data, other.means.data], dim=0)
        opacities = torch.cat([self.opacities.data, other.opacities.data], dim=0)
        scales = torch.cat([self.scales.data, other.scales.data], dim=0)
        quats = torch.cat([self.quats.data, other.quats.data], dim=0)
        others = (
            {key: torch.cat([getattr(self, key).data, getattr(other, key).data], dim=0) for key in self._other_keys}
            if hasattr(self, "_other_keys")
            else None
        )
        return Gaussians(
            means,
            opacities=opacities,
            scales=scales,
            quats=quats,
            other_attrs=others,
        )

    def detach(self):
        means = self.means.data.detach()
        opacities = self.opacities.data.detach()
        scales = self.scales.data.detach()
        quats = self.quats.data.detach()
        others = (
            {key: getattr(self, key).data.detach() for key in self._other_keys}
            if hasattr(self, "_other_keys")
            else None
        )
        return Gaussians(
            means,
            opacities=opacities,
            scales=scales,
            quats=quats,
            other_attrs=others,
        )

    @classmethod
    def empty_like(cls, gaussians: Gaussians) -> Gaussians:
        """Create an empty instance of Gaussians with the same feature shapes and attributes as the input."""
        means = torch.empty((0, 3), dtype=gaussians.means.dtype, device=gaussians.means.device)
        opacities = torch.empty((0, 1), dtype=gaussians.opacities.dtype, device=gaussians.opacities.device)
        scales = torch.empty((0, 3), dtype=gaussians.scales.dtype, device=gaussians.scales.device)
        quats = torch.empty((0, 4), dtype=gaussians.quats.dtype, device=gaussians.quats.device)
        others = (
            {
                key: torch.empty(
                    (0, getattr(gaussians, key).shape[1]) if len(getattr(gaussians, key).shape) > 1 else (0,),
                    dtype=getattr(gaussians, key).dtype,
                    device=getattr(gaussians, key).device,
                )
                for key in gaussians._other_keys
            }
            if hasattr(gaussians, "_other_keys")
            else None
        )
        return Gaussians(
            means,
            opacities=opacities,
            scales=scales,
            quats=quats,
            other_attrs=others,
        )

    def __getitem__(self, idx):
        """Gaussian slicing. NOTE: This function breaks gradient flow!"""
        means = self.means.data[idx]
        opacities = self.opacities.data[idx]
        scales = self.scales.data[idx]
        quats = self.quats.data[idx]
        others = (
            {key: getattr(self, key).data[idx] for key in self._other_keys} if hasattr(self, "_other_keys") else None
        )
        return Gaussians(
            means,
            opacities=opacities,
            scales=scales,
            quats=quats,
            other_attrs=others,
        )

    def to_dict(self):
        """Returns a dictionary representation of the Gaussians object."""
        attributes = {
            "means": self.means,
            "opacities": self.opacities,
            "scales": self.scales,
            "quats": self.quats,
        }
        attributes.update({key: getattr(self, key) for key in self._other_keys} if hasattr(self, "_other_keys") else {})
        return attributes

    def __repr__(self) -> str:
        """Returns a dictionary like visualization of all attributes of the Gaussians object.

        Iterates through all attributes of the Gaussians object and returns a string representation of the attributes.
        """
        attributes = self.to_dict()
        return (
            "Gaussians(" + ", ".join([f"{k}={v.data if v is not None else None}" for k, v in attributes.items()]) + ")"
        )
