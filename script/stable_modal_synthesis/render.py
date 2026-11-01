"""Continuous float64 synthesis using the training solver's forward step."""

import ctypes as ct
import time

import torch

from . import reference as ref
from .native import accelerate, call, doubles
from .training import model_arrays, parameters, synthesis_inputs


class Stream:
    """Snapshot a model and retain q, p, psi and the projection cache across blocks.

    Each output sample is the state before consuming the corresponding excitation.
    Use one stream per independent trajectory; calls on a stream must be sequential.
    """

    def __init__(self, ps, batch):
        self.handle = ct.c_void_p()
        initial = batch["initial"]
        if initial.ndim != 2 or initial.shape[1] not in (150, 151) or not 1 <= len(initial) <= 2048:
            raise ValueError("Expected 1..2048 states containing 75 q,p pairs and optional psi")
        self.count = len(initial)
        if batch["physical"].shape != (self.count, 226):
            raise ValueError("Expected omega squared, sigma, excitation shape and mu per state")
        arrays = model_arrays(ps)[:3] + [doubles(batch["physical"]), doubles(initial)]
        self.lib = accelerate()
        call(
            self.lib.modal_stream_create,
            len(ps[1]),
            self.count,
            initial.shape[1],
            batch["fs"],
            batch["lambda0"],
            *arrays,
            ct.byref(self.handle)
        )

    def process(self, excitation):
        """Consume (bodies, frames) excitation and return (bodies, 151, frames) states."""
        if not self.handle.value:
            raise RuntimeError("Closed Accelerate renderer")
        if excitation.ndim != 2 or excitation.shape[0] != self.count or excitation.shape[1] > 2**32 - 1:
            raise ValueError("Expected one excitation row per body")
        frames = excitation.shape[1]
        output = torch.empty((self.count, frames, 151), dtype=torch.float64)
        excitation = doubles(excitation)
        call(self.lib.modal_stream_process, self.handle, frames, excitation, output)
        return output.movedim(-1, 1)

    def close(self):
        if self.handle.value:
            self.lib.modal_stream_destroy(self.handle)
            self.handle.value = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


@torch.no_grad()
def render(model, batch):
    """Render a complete author-format recording with bounded solver scratch space."""
    prepared = synthesis_inputs(model, batch)
    start = time.perf_counter()
    with Stream(parameters(model), prepared) as stream:
        output = stream.process(prepared["excitation"])
    audio = ref.pickup(model, batch, output)
    return dict(output=output, w=audio, timing=dict(wall_seconds=time.perf_counter() - start))
