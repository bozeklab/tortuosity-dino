import numpy as np
from torch import nn


def clip_gradients(
        model: nn.Module,
        clip: float=3.0
):
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)


def cosine_scheduler(
        base_value: float,
        final_value: float,
        epochs: int,
        niter_per_ep: int,
        warmup_epochs: int=0,
        start_warmup_value: float=0.
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep, f"{len(schedule)} vs {epochs * niter_per_ep}"
    return schedule


def cancel_gradients_layer_x(
        epoch: int,
        model: nn.Module,
        freeze_duration: int,
        layer_name: str
):
    if epoch >= freeze_duration:
        return
    for n, p in model.named_parameters():
        if layer_name in n:
            p.grad = None
