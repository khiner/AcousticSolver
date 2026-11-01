"""Local run provenance and resumable optimization."""

import hashlib
import importlib.metadata
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch

from . import UPSTREAM

STOP = False


def interrupt(signum, frame):
    global STOP
    STOP = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, config, sources, resume):
    path = output / "run.json"
    source_hashes = {str(p.resolve()): digest(p) for p in sources}
    implementation = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    if resume:
        previous = json.loads(path.read_text())
        if (
            previous["config"] != config
            or previous["sources"] != source_hashes
            or previous["implementation"] != implementation
        ):
            raise ValueError("Resume requires unchanged configuration, inputs and implementation")
    else:
        output.mkdir(parents=True, exist_ok=False)
        path.write_text(
            json.dumps(
                dict(
                    upstream_commit=UPSTREAM,
                    config=config,
                    sources=source_hashes,
                    implementation=implementation,
                    dependencies={
                        p: importlib.metadata.version(p)
                        for p in ("torch", "torchaudio", "numpy", "scipy", "geomloss", "meshio", "open3d")
                    },
                    complete_paper_reproduction=False,
                    backend="Metal float32 audio/neural SDF; CPU float64 FEM and SciPy ARPACK",
                    differences=[
                        "CPU/MPS random initialization is seeded locally; author seeds are not supplied",
                        "FEM quadrature accumulation and eigenvalue sensitivities use float64",
                        "Transport coordinates preserve upstream duplicate-index gradient accumulation",
                        "Shape, thickness and morphing use eigenvalue supervision, as in released scripts",
                        "Material inference assumes a known impulse; this is not joint contact-force recovery",
                    ],
                ),
                indent=2,
            )
            + "\n"
        )
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)


def train(output, name, model, steps, closure, optimizer, scheduler=None, reset=None):
    output.mkdir(parents=True, exist_ok=True)
    path = output / (name + ".pt")
    state = dict(step=0, extra={}, history=[], elapsed_seconds=0.0)
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        if reset and state["step"] > reset[0]:
            optimizer, scheduler = reset[1]()
        optimizer.load_state_dict(state["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        torch.mps.set_rng_state(state["mps_rng"])
    initial_elapsed = state["elapsed_seconds"]
    start = time.monotonic()

    def save():
        state.update(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler else None,
            rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state(),
            elapsed_seconds=initial_elapsed + time.monotonic() - start,
        )
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    while state["step"] < steps and not state.get("converged", False) and not STOP:
        i = state["step"]
        if reset and i == reset[0]:
            optimizer, scheduler = reset[1]()
        optimizer.zero_grad()
        loss, metrics = closure(i, state["extra"])
        if loss is None:
            state["converged"] = True
            break
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name}: nonfinite loss at step {i}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"{name}: nonfinite gradient at step {i}")
        optimizer.step()
        if scheduler:
            scheduler.step()
        state["step"] += 1
        if i == 0 or state["step"] % 25 == 0 or state["step"] == steps:
            row = dict(step=state["step"], loss=float(loss.detach()), **metrics)
            state["history"].append(row)
            save()
            print(json.dumps(dict(stage=name, elapsed_seconds=state["elapsed_seconds"], **row)), flush=True)
    save()
    if STOP:
        raise SystemExit("Stopped after saving the current update")
    return state


def save_result(output, report, **arrays):
    np.savez_compressed(
        output / "result.npz",
        **{k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in arrays.items()},
    )
    report["artifacts"] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "result.json"}
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
