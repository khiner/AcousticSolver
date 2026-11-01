"""Spectral string generation with Metal float-pair state and arithmetic."""

import ctypes as ct

import torch

from . import reference as ref
from .native import SOURCE, DOUBLE, call, doubles, library

SHADER_SOURCES = (ref.ROOT / "src/Metal/FloatPair.h", SOURCE / "Modal.metal")


class Renderer:
    def __init__(self):
        self.lib = library("StableModalMetal")
        self.lib.modal_create.argtypes = [ct.c_char_p, ct.c_void_p, ct.c_size_t]
        self.lib.modal_create.restype = ct.c_void_p
        self.lib.modal_destroy.argtypes = [ct.c_void_p]
        self.lib.modal_destroy.restype = None
        self.lib.modal_render.argtypes = [
            ct.c_void_p,
            *([ct.c_uint32] * 5),
            ct.c_double,
            ct.c_double,
            *([DOUBLE] * 6),
            ct.c_void_p,
            ct.c_size_t,
        ]
        self.lib.modal_render.restype = ct.c_int
        error = ct.create_string_buffer(8192)
        source = "\n".join(path.read_text() for path in SHADER_SOURCES)
        self.handle = self.lib.modal_create(source.encode(), error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode())

    def close(self):
        if self.handle:
            self.lib.modal_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def render(self, model, batch, chunk=64):
        m = batch["omega"].shape[-1]
        nl = model.nl
        from torch_dct import idct

        weights = idct(torch.eye(m + 1, dtype=torch.float64), norm="ortho")[1:].T * (m + 1) ** 0.5 * nl._beta
        fs, frames, shape, force = ref.prepared_inputs(model, batch)
        physical = torch.cat((batch["omega"].square(), batch["sigma"], shape, batch["mu"][:, None]), -1)
        initial = batch["y0"]
        if initial.shape[-1] == 2 * m:
            initial = torch.cat((initial, torch.sqrt(2 * nl.potential(initial[:, :m]) + 1e-12)[:, None]), -1)
        arrays = [doubles(x) for x in (weights, physical, force, initial)]
        count = initial.shape[0]
        output = torch.empty((count, frames, 2 * m + 1), dtype=torch.float64)
        seconds = torch.empty(2, dtype=torch.float64)
        call(
            self.lib.modal_render,
            self.handle,
            m,
            len(weights),
            count,
            frames,
            chunk,
            fs,
            model._solver._method._lambda0,
            *arrays,
            output,
            seconds
        )
        state = output.movedim(-1, 1)
        audio = ref.pickup(model, batch, state)
        return dict(output=state, w=audio, timing=dict(wall_seconds=seconds[0].item(), gpu_seconds=seconds[1].item()))
