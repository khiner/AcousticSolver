"""Shared Accelerate bindings, SAV training and full-batch gradient accumulation."""

import torch

from . import reference as ref
from .native import accelerate, call, doubles

PARAMETERS = ("_weight", "_bias", "_log_scale_in", "_log_scale_out")


def parameters(model):
    return [getattr(model.nl, name) for name in PARAMETERS]


def flatten(values):
    return torch.cat([x.detach().cpu().reshape(-1) for x in values])


@torch.no_grad()
def synthesis_inputs(model, batch):
    fs, _, phi, excitation = ref.prepared_inputs(model, batch)
    method = model._solver._method
    if method._eps != 1e-12 or model.nl._activation.negative_slope != 0.01:
        raise ValueError("Expected released SAV epsilon and leaky-ReLU slope")
    return dict(
        fs=fs,
        lambda0=method._lambda0,
        initial=batch["y0"],
        excitation=excitation,
        physical=torch.cat((batch["omega"].square(), batch["sigma"], phi, batch["mu"][:, None]), -1),
    )


def training_inputs(model, batch):
    return dict(synthesis_inputs(model, batch), target=batch["target_output"])


@torch.no_grad()
def model_arrays(ps):
    if len(ps) != 4 or any(p.dtype != torch.float64 for p in ps):
        raise ValueError("Expected four float64 model parameter arrays")
    w, bias, log_in, log_out = ps
    if (
        bias.ndim != 1
        or not 1 <= len(bias) <= 1000
        or w.shape != (len(bias), 75)
        or log_in.shape != bias.shape
        or log_out.shape != bias.shape
    ):
        raise ValueError("Expected the released GradNet parameter layout")
    scale_in = log_in.exp()
    return [doubles(x) for x in (scale_in[:, None] * w, bias, log_out.exp(), scale_in)]


class Trainer:
    def __init__(self):
        self.lib = accelerate()

    @torch.no_grad()
    def evaluate(self, ps, batch, backward=True):
        if batch["initial"].ndim != 2 or batch["initial"].shape[1] != 150:
            raise ValueError("Training requires 75 modes and teacher-forced q,p initial conditions")
        fs, frames, excitation = batch["fs"], batch["target"].shape[-1], batch["excitation"]
        count = len(batch["initial"])
        if not 1 <= count <= 2048 or not 2 <= frames <= 128:
            raise ValueError("Expected 1..2048 windows of 2..128 samples")
        for name, shape in dict(physical=(count, 226), excitation=(count, frames), target=(count, 151, frames)).items():
            if batch[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        arrays = model_arrays(ps) + [
            doubles(x) for x in (batch["physical"], excitation, batch["initial"], batch["target"].movedim(1, -1))
        ]
        output = torch.empty((count, frames, 151), dtype=torch.float64)
        gradient = torch.empty(sum(p.numel() for p in ps), dtype=torch.float64) if backward else None
        initial_gradient = torch.empty((count, 150), dtype=torch.float64) if backward else None
        loss = torch.empty(1, dtype=torch.float64)
        call(
            self.lib.modal_evaluate,
            len(ps[1]),
            count,
            frames,
            fs,
            batch["lambda0"],
            *arrays,
            output,
            gradient,
            initial_gradient,
            loss,
        )
        return dict(
            output=output.movedim(-1, 1), gradient=gradient, initial_gradient=initial_gradient, loss=loss[0].item()
        )

    @torch.no_grad()
    def accumulate(self, ps, batch, chunk=256, backward=True):
        """Accumulate a mean loss and its gradient without changing parameters between chunks."""
        if not 1 <= chunk <= 2048:
            raise ValueError("Expected a chunk size of 1..2048 windows")
        count = len(batch["initial"])
        if not count:
            raise ValueError("Expected a nonempty batch")
        gradient = torch.zeros(sum(p.numel() for p in ps), dtype=torch.float64) if backward else None
        loss = 0.0
        for start in range(0, count, chunk):
            end = min(start + chunk, count)
            part = {k: v[start:end] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            result = self.evaluate(ps, part, backward)
            weight = (end - start) / count
            if backward:
                gradient.add_(result["gradient"], alpha=weight)
            loss += weight * result["loss"]
        return dict(loss=loss, gradient=gradient)

    @torch.no_grad()
    def adam(self, ps, gradient, state, lr=0.001, betas=(0.9, 0.999), epsilon=1e-8):
        """Update raw parameters and moments, including restored checkpoint moments."""
        raw = doubles(flatten(ps))
        grad = doubles(gradient)
        first, second = state["first"], state["second"]
        if any(
            x.shape != raw.shape or x.dtype != torch.float64 or x.device.type != "cpu" or not x.is_contiguous()
            for x in (grad, first, second)
        ):
            raise ValueError("Adam buffers must be contiguous float64 vectors with the parameter size")
        step = state["step"] + 1
        call(self.lib.modal_adam, len(raw), step, lr, *betas, epsilon, raw, grad, first, second)
        for p, value in zip(ps, raw.split([p.numel() for p in ps])):
            p.copy_(value.view_as(p))
        state["step"] = step
