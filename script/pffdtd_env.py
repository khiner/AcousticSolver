"""Shared plumbing for the scripts that drive the PFFDTD reference.

Where things live, how to get into the virtualenv that runs the reference's Python, and
the compatibility patches that let a 2021 codebase import under a current interpreter.
Those patches live here rather than in ../pffdtd so the checkout stays a plain checkout,
byte-comparable with upstream, and the goldens it produces stay the reference's own
arithmetic.
"""

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFFDTD_DIR = os.environ.get("PFFDTD_DIR", os.path.join(os.path.dirname(ROOT), "pffdtd"))
VENV_DIR = os.path.join(ROOT, "build", "venv-pffdtd")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python")
ENGINE_DIR = os.path.join(ROOT, "build", "pffdtd-engine")
SIM_DIR = os.path.join(ROOT, "build", "pffdtd")
SCENES_DIR = os.path.join(ROOT, "Scenes")
GEN_DIR = os.path.join(ROOT, "gen", "room")

# What the reference's sim_setup needs. Its pip_requirements also lists the drawing stack
# (mayavi, vtk, polyscope) and the output post-processing (matplotlib, resampy), neither of
# which we run — we post-process the raw receiver signals ourselves.
REQUIREMENTS = ["numpy", "scipy", "numba", "h5py", "tqdm", "psutil", "memory_profiler"]


def ensure_venv(script):
    """Re-exec `script` inside build/venv-pffdtd, creating the venv on first run."""
    if os.path.abspath(sys.prefix) == os.path.abspath(VENV_DIR):
        return
    if not os.path.isfile(VENV_PYTHON):
        print(f"==> Creating {VENV_DIR}")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
        subprocess.run([VENV_PYTHON, "-m", "pip", "-q", "install", "--upgrade", "pip", "wheel"], check=True)
        subprocess.run([VENV_PYTHON, "-m", "pip", "-q", "install", *REQUIREMENTS], check=True)
    os.execv(VENV_PYTHON, [VENV_PYTHON, os.path.abspath(script), *sys.argv[1:]])


def require_checkout():
    if not os.path.isdir(os.path.join(PFFDTD_DIR, "python")):
        sys.exit(f"no PFFDTD checkout at {PFFDTD_DIR} — clone https://github.com/bsxfun/pffdtd there")


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROGRESS_PREFIXES = ("Running [", "T: ", "I: ", "TA: ", "TB: ", "TPW: ")


def filter_log(raw_path, log_path):
    """Rewrite an engine log without its per-timestep progress redraw.

    Both engines repaint a six-line block with the cursor-up escape every single timestep,
    so a long run leaves hundreds of megabytes of escape codes around a few hundred lines
    that actually say something. Those lines are what is worth keeping and committing.
    """
    kept = []
    with open(raw_path, errors="replace") as raw:
        for line in raw:
            line = ANSI.sub("", line).rstrip() + "\n"
            if line.strip() and not line.startswith(PROGRESS_PREFIXES):
                kept.append(line)
    with open(log_path, "w") as out:
        out.writelines(kept)
    return kept


def collect_receivers(sim_dir, outs_path, out_path):
    """Fold an engine's sim_outs.h5 into one float64 row per receiver, and write it out.

    Every engine writes one row per trilinear corner, eight per receiver, in the order the
    setup wrote out_ixyz. Those corners recombine with the weights the setup also wrote.
    Nothing else is applied: the integrator and low-cut that PFFDTD's own post-processing
    adds are common to both sides of any comparison and only mask differences.
    """
    import h5py
    import numpy as np

    with h5py.File(os.path.join(sim_dir, "comms_out.h5"), "r") as f:
        out_alpha = f["out_alpha"][...]
    with h5py.File(outs_path, "r") as f:
        u_out = f["u_out"][...]
    r_out = np.sum((u_out * out_alpha.reshape(-1)[:, None]).reshape((*out_alpha.shape, -1)), axis=1)
    np.ascontiguousarray(r_out, dtype=np.float64).tofile(out_path)
    return r_out.shape


def sample_rate(sim_dir):
    import h5py

    with h5py.File(os.path.join(sim_dir, "sim_consts.h5"), "r") as f:
        return float(f["SR"][()])


def pffdtd_revision():
    result = subprocess.run(
        ["git", "-C", PFFDTD_DIR, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip()


def record_reference(out_dir, sim_dir, stem, entry):
    """Merge one engine's run into gen/room/<Scene>/reference.json, keeping the other engines'.

    Every engine's outputs land beside each other under one scene, and the rate and the
    reference revision belong to the scene rather than to any one of them.
    """
    path = os.path.join(out_dir, "reference.json")
    meta = json.load(open(path)) if os.path.isfile(path) else {}
    meta[stem] = entry
    meta["srate"] = sample_rate(sim_dir)
    meta["pffdtd"] = pffdtd_revision()
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def apply_compat():
    """Patch the interpreter so the reference imports and runs unmodified."""
    import multiprocessing
    from multiprocessing import shared_memory

    import numpy as np

    # Importing the reference would otherwise leave __pycache__ trees all over the checkout.
    sys.dont_write_bytecode = True

    # The voxelizer hands its workers a closure, which only a forking start method can
    # carry. macOS has defaulted to spawn since 3.8.
    multiprocessing.set_start_method("fork", force=True)

    # Python 3.12 made mmap.close() raise while any buffer view is still alive. The
    # voxelizer closes its shared segments with NumPy views over them still in scope, and
    # only after everything has been read out, so letting the close slide loses nothing.
    close = shared_memory.SharedMemory.close

    def close_ignoring_live_views(self):
        try:
            close(self)
        except BufferError:
            pass

    shared_memory.SharedMemory.close = close_ignoring_live_views

    # Aliases NumPy removed in 1.24 and 2.0 that the reference still reaches for.
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "bool"):
        np.bool = bool
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_

    sys.path.insert(0, os.path.join(PFFDTD_DIR, "python"))
