"""Custom Optimizers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

import torch
from nerfstudio.configs import base_config


@dataclass
class SGDOptimizerConfig(base_config.PrintableConfig):
    """Configuration for the SGD optimizer."""

    _target: Type = torch.optim.SGD
    lr: float = 0.01
    """The learning rate."""
    momentum: float = 0.9
    """The momentum factor."""
    weight_decay: float = 0.0001
    """The weight decay factor."""
    nesterov: bool = False
    """Whether to use Nesterov momentum."""
    max_norm: Optional[float] = None
    """The max norm to use for gradient clipping."""

    def setup(self, params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        kwargs.pop("max_norm")
        return self._target(params, **kwargs)
