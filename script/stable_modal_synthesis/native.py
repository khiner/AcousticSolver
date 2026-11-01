"""Load the CMake-built native libraries and call their C interfaces."""

import ctypes as ct
from functools import cache
from pathlib import Path

import torch

from . import reference as ref

SOURCE = ref.ROOT / "src/StableModal"
SCRIPT = Path(__file__).parent
DOUBLE = ct.POINTER(ct.c_double)


def library(name):
    path = ref.ROOT / "build" / f"lib{name}.dylib"
    if not path.is_file():
        raise FileNotFoundError(f"Build the native libraries with: cmake --build build --target {name}")
    return ct.CDLL(str(path))


def doubles(tensor):
    return tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()


def call(function, *args):
    converted = [ct.cast(a.data_ptr(), DOUBLE) if isinstance(a, torch.Tensor) else a for a in args]
    error = ct.create_string_buffer(8192)
    if function(*converted, error, len(error)):
        raise RuntimeError(error.value.decode())


@cache
def accelerate():
    lib = library("StableModalTraining")
    lib.modal_evaluate.argtypes = [
        *([ct.c_uint32] * 3),
        ct.c_double,
        ct.c_double,
        *([DOUBLE] * 12),
        ct.c_void_p,
        ct.c_size_t,
    ]
    lib.modal_evaluate.restype = ct.c_int
    lib.modal_adam.argtypes = [
        ct.c_uint32,
        ct.c_uint32,
        *([ct.c_double] * 4),
        *([DOUBLE] * 4),
        ct.c_void_p,
        ct.c_size_t,
    ]
    lib.modal_adam.restype = ct.c_int
    lib.modal_training_memory.argtypes = [ct.POINTER(ct.c_uint64)]
    lib.modal_training_memory.restype = ct.c_int

    lib.modal_stream_create.argtypes = [
        *([ct.c_uint32] * 3),
        ct.c_double,
        ct.c_double,
        *([DOUBLE] * 5),
        ct.POINTER(ct.c_void_p),
        ct.c_void_p,
        ct.c_size_t,
    ]
    lib.modal_stream_create.restype = ct.c_int
    lib.modal_stream_process.argtypes = [ct.c_void_p, ct.c_uint32, DOUBLE, DOUBLE, ct.c_void_p, ct.c_size_t]
    lib.modal_stream_process.restype = ct.c_int
    lib.modal_stream_destroy.argtypes = [ct.c_void_p]
    lib.modal_stream_destroy.restype = None
    return lib


def memory():
    values = (ct.c_uint64 * 2)()
    if accelerate().modal_training_memory(values):
        raise RuntimeError("Failed to read process footprint")
    return dict(zip(("physical_footprint_bytes", "peak_physical_footprint_bytes"), values))
