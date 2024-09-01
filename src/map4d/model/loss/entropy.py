import torch
from torch import Tensor


def compute_entropy(composite_weights: Tensor, entropy_skewness=1.0, eps: float = 1e-6) -> Tensor:
    """Compute entropy of the composite weights.

    Args:
        composite_weights (Tensor): N_rays, N_samples, 2
        entropy_skewness (float, optional): skewness parameter (cf. banmo paper). Defaults to 1.0.
        eps (float, optional): numerical stability parameter. Defaults to 1e-6.

    Returns:
        Tensor: Entropy per pixel.
    """
    # calculate dynamic density ratio along rays
    weights_ratio = composite_weights[..., 1].pow(entropy_skewness)
    inv_weights_ratio = 1.0 - weights_ratio
    # supervise integral (n-sum interpolated) of entropy along rays
    entropy = -(
        weights_ratio * torch.log(weights_ratio + eps) + inv_weights_ratio * torch.log(inv_weights_ratio + eps)
    ).sum(dim=-1, keepdim=True)
    return entropy
