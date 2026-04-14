"""
SDE baseline utilities.
"""
import math

import torch


def _forward_with_cfg(model, x, t, y, cfg_scale):
    n = x.shape[0]
    x_cfg = torch.cat([x, x], dim=0)
    y_null = torch.full((n,), 1000, device=y.device, dtype=torch.long)
    y_cfg = torch.cat([y, y_null], dim=0)
    t_cfg = torch.cat([t, t], dim=0)
    out_cfg = model.forward_with_cfg(x_cfg, t_cfg, y_cfg, cfg_scale)
    return out_cfg[:n]


def sde_1step_logprob(model, z, x1, y, cfg_scale=1.5, sigma=0.1):
    """
    One-step SDE log probability approximation.

    IMPORTANT: dt = -1.0, so this Gaussian approximation has O(1) bias.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    t = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
    v = _forward_with_cfg(model, z, t, y, cfg_scale)
    mu = z + v * (-1.0)
    std = sigma

    diff = x1 - mu
    dim = diff[0].numel()
    quad = diff.flatten(start_dim=1).pow(2).sum(dim=1) / (std ** 2)
    const = dim * math.log(2.0 * math.pi * (std ** 2))
    log_prob = -0.5 * (quad + const)
    return log_prob
