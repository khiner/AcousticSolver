"""Optimize image-guided acoustic reliefs on Apple GPUs with resumable checkpoints."""

import argparse
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import tempfile
import time
import tomllib

import cv2
import drjit as dr
import mitsuba as mi
import numpy as np
from PIL import Image
import torch

from .appearance import Appearance, CAMERAS, ImageLoss
from .geometry import paper_mesh
from .native import Acoustic, evaluate
from .regularizers import (
    barrier_loss,
    neg_relu,
    normalize_gradients,
    preprocess_tex,
    smoothness,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def listeners(radius, spacing):
    angles = [(0, 0)] + [
        (e, a)
        for e in np.arange(spacing, 91, spacing)
        for a in np.arange(0, 360, spacing)
    ]
    e, a = np.radians(np.asarray(angles)).T
    return (
        radius
        * np.column_stack([np.sin(e) * np.cos(a), np.cos(e), -np.sin(e) * np.sin(a)])
    ).tolist()


def write_mesh(output, problem, pixels, solver):
    with tempfile.TemporaryDirectory(prefix="mesh-", dir=output) as directory:
        source = Path(directory)
        vertices = evaluate(
            problem, pixels, source / "input.json", source / "result.json", solver, True
        )["vertices"]
    with (output / "mesh.obj").open("w") as file:
        for x, y, z in vertices:
            file.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in problem["triangles"]:
            file.write(f"f {a + 1} {b + 1} {c + 1}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--steps", type=int, help="Total completed updates to reach in this invocation"
    )
    parser.add_argument(
        "--refine",
        type=Path,
        help="Completed first-stage output to refine for fifty updates",
    )
    parser.add_argument("--solver", type=Path, default=Path("build/BemTextureSolve"))
    a = parser.parse_args()
    config = tomllib.loads(a.config.read_text())
    options, image_options, bem_options = (
        config[k] for k in ["optimization", "image", "diffbem"]
    )
    if (
        options.get("guidance_type", "image") != "image"
        or not options.get("normalize_grads", True)
        or options.get("vmax", 0) <= 0
    ):
        parser.error(
            "Expected image guidance, normalized gradients and a positive height bound"
        )
    if bem_options.get("n_freqs", 1) != 1 or not bem_options.get("sample_freq", False):
        parser.error(
            "The image-guided driver requires one sampled frequency per update"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("The appearance objective requires MPS")
    maximum = 50 if a.refine else options["iters"]
    target_steps = a.steps if a.steps is not None else maximum
    if not 1 <= target_steps <= maximum:
        parser.error(f"Need 1..{maximum} updates")
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20260908)
    torch.mps.manual_seed(20260908)
    height_limit, border = float(options["vmax"]), int(image_options["edge_border"])
    size = int(image_options["hfield_res"])
    if not 0 < border < size / 2:
        parser.error("Expected a nonempty texture interior and border")
    if a.refine:
        prior = np.load(a.refine / "hfield.npy", allow_pickle=False)
        interior = prior[border:-border, border:-border]
        initial = cv2.resize(interior, (interior.shape[1] * 2, interior.shape[0] * 2))
        size = initial.shape[0] + 2 * border
    else:
        initial = np.full(
            (size - 2 * border, size - 2 * border),
            options.get("init_val", 0),
            dtype=np.float32,
        )
    if not np.isfinite(initial).all() or np.max(np.abs(initial)) > height_limit:
        raise ValueError("Initial texture exceeds the finite height bound")
    sources = [
        Path(__file__).with_name(name + ".py")
        for name in [
            "run",
            "appearance",
            "resize",
            "regularizers",
            "geometry",
            "native",
        ]
    ]
    loss_model = ImageLoss()
    sources += [
        a.solver,
        Path("src/Bem/BemKernels.metal"),
        Path("src/Bem/BemParams.h"),
        Path("build/reliefs-models/ViT-B-32.pt"),
        a.config,
        a.target,
    ]
    if a.refine:
        sources += [a.refine / "hfield.npy"]
    manifest = dict(
        config=config,
        stage=2 if a.refine else 1,
        seed=20260908,
        frequency_seed=42,
        size=size,
        torch=torch.__version__,
        numpy=np.__version__,
        mitsuba=mi.__version__,
        drjit=dr.__version__,
        dependencies={
            name: version(name)
            for name in [
                "torchvision",
                "pillow",
                "opencv-python-headless",
                "gmsh",
                "clip",
            ]
        },
        clip_source=json.loads(distribution("clip").read_text("direct_url.json")),
        renderer="metal_ad_rgb",
        clip_device="mps",
        clip_dtype="float32",
        adam_foreach=False,
        sha256={str(p.resolve()): digest(p) for p in sources},
    )
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest["problem_sha256"] = digest(out / "problem.json")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run inputs, mesh, dependencies or sources changed")
        problem = json.loads((out / "problem.json").read_text())
    else:
        points, faces, selected = paper_mesh(config["dimensions"], int(bool(a.refine)))
        problem = dict(
            vertices=points.tolist(),
            triangles=faces.tolist(),
            height_vertices=selected.tolist(),
            source=bem_options.get("src_pt", [0, 100, 0]),
            listeners=listeners(
                bem_options.get("listener_radius", 50),
                bem_options.get("listener_ds", 5),
            ),
            sound_speed=344,
            frequency_hz=bem_options["freq_bands"][0],
        )
        save(out / "problem.json", problem)
        manifest["problem_sha256"] = digest(out / "problem.json")
        save(manifest_path, manifest)
    frequencies = np.asarray(bem_options["freq_bands"], dtype=float)
    frequency_weights = 1 / frequencies
    schedule = np.random.RandomState(42).choice(
        frequencies, maximum, p=frequency_weights / frequency_weights.sum()
    )
    scene = Appearance(problem["vertices"], problem["triangles"], a.target, loss_model)
    acoustic = Acoustic(problem, out / "acoustic", a.solver)
    with Image.open(a.target) as picture:
        target = (
            np.asarray(
                picture.resize(
                    (size, size), resample=Image.Resampling.BILINEAR
                ).convert("L")
            )
            / 255
        )
    target = torch.tensor(
        np.stack([target, target, target], axis=0)[None],
        device="mps",
        dtype=torch.float32,
    )
    x = torch.tensor(initial, device="mps", requires_grad=True)
    anchor = preprocess_tex(x.detach().clone(), border)
    optimizer = torch.optim.Adam(
        [x], lr=1e-3 if a.refine else options.get("lr", 1e-3), foreach=False
    )
    pointer = out / "checkpoint.json"
    start = 0
    if pointer.exists():
        saved = json.loads(pointer.read_text())
        path = out / saved["file"]
        if digest(path) != saved["sha256"]:
            raise ValueError("Checkpoint digest mismatch")
        with np.load(path, allow_pickle=False) as state:
            start = int(state["iteration"])
            if start != saved["iteration"]:
                raise ValueError("Checkpoint iteration mismatch")
            with torch.no_grad():
                x.copy_(torch.tensor(state["pixels"], device="mps"))
            optimizer.state[x] = dict(
                step=torch.tensor(float(start)),
                exp_avg=torch.tensor(state["moment"], device="mps"),
                exp_avg_sq=torch.tensor(state["variance"], device="mps"),
            )
            torch.set_rng_state(torch.from_numpy(state["torch_rng"].copy()))
            torch.mps.set_rng_state(torch.from_numpy(state["mps_rng"].copy()))
    if start > target_steps:
        raise ValueError("Target precedes completed updates")
    weights = dict(
        ac=0 if a.refine else options.get("ac_wt", 5),
        cl=options.get("cl_wt", 3),
        sm=options.get("sm_wt", 10),
        ba=options.get("ba_wt", 0.5),
        ng=options.get("ng_wt", 0.5),
    )
    if a.refine:
        weights["rg"] = 5
    for i, weight in enumerate(options.get("vw_wts", [3, 1, 1, 1, 1])):
        weights[f"vw_{i}"] = weight
    if len([key for key in weights if key.startswith("vw_")]) != 5:
        raise ValueError("Expected five view weights")
    resolution = 512 if a.refine else 256
    for iteration in range(start, target_steps):
        began = time.monotonic()
        optimizer.zero_grad()
        full = preprocess_tex(x, border)
        pixels = full.detach().cpu().numpy()
        values, gradients, audits = {}, {}, {}
        frequency = float(schedule[iteration])
        print(f"Step {iteration + 1}/{maximum}, {frequency:g} Hz", flush=True)
        if weights["ac"]:
            values["ac"], gradient, result = acoustic.gradient(pixels, frequency)
            gradients["ac"] = torch.tensor(
                gradient[border:-border, border:-border], device="mps"
            )
            audits = {
                key: result[key]
                for key in [
                    "diffusion",
                    "forward_residual",
                    "adjoint_residual",
                    "independent_forward_residual",
                    "independent_adjoint_residual",
                    "forward_iterations",
                    "adjoint_iterations",
                ]
            }
        else:
            values["ac"], gradients["ac"] = 0, torch.zeros_like(x)
        for index, camera in enumerate(CAMERAS):
            value, gradient = scene.gradient(
                pixels,
                *camera,
                radius=image_options.get("cam_rad", 1),
                resolution=resolution,
            )
            values[f"vw_{index}"] = value
            gradients[f"vw_{index}"] = torch.tensor(
                gradient[border:-border, border:-border], device="mps"
            )
        values["cl"], gradient = loss_model.texture(pixels, target)
        gradients["cl"] = torch.tensor(
            gradient[border:-border, border:-border], device="mps"
        )
        terms = dict(
            sm=smoothness(full), ba=barrier_loss(full, height_limit), ng=neg_relu(full)
        )
        if a.refine:
            terms["rg"] = torch.mean((full - anchor) ** 2)
        for key, value in terms.items():
            values[key] = float(value.detach().cpu())
            gradients[key] = torch.autograd.grad(value, x, retain_graph=True)[0]
        total = torch.zeros_like(x)
        for key, weight in weights.items():
            total += weight * normalize_gradients(gradients[key])
        if not torch.isfinite(total).all() or not all(
            np.isfinite(v) for v in values.values()
        ):
            raise ValueError("Nonfinite objective or gradient")
        x.grad = total
        optimizer.step()
        with torch.no_grad():
            x.clamp_(-height_limit, height_limit)
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite updated texture")
        state = optimizer.state[x]
        path = checkpoints / f"step-{iteration + 1:04}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                iteration=iteration + 1,
                pixels=x.detach().cpu().numpy(),
                moment=state["exp_avg"].cpu().numpy(),
                variance=state["exp_avg_sq"].cpu().numpy(),
                torch_rng=torch.get_rng_state().numpy(),
                mps_rng=torch.mps.get_rng_state().numpy(),
            )
        temporary.replace(path)
        row = dict(
            iteration=iteration + 1,
            frequency_hz=frequency,
            loss=sum(weights[key] * values[key] for key in weights),
            components=values,
            gradient_norm=float(total.norm().cpu()),
            height_min=float(x.min().detach().cpu()),
            height_max=float(x.max().detach().cpu()),
            seconds=time.monotonic() - began,
            acoustic=audits,
        )
        save(checkpoints / f"step-{iteration + 1:04}.json", row)
        save(
            pointer,
            dict(
                iteration=iteration + 1,
                file=str(path.relative_to(out)),
                sha256=digest(path),
            ),
        )
        print(json.dumps(row), flush=True)
    full = preprocess_tex(x, border).detach().cpu().numpy()
    with (out / "hfield.tmp").open("wb") as file:
        np.save(file, full, allow_pickle=False)
    (out / "hfield.tmp").replace(out / "hfield.npy")
    rows = [
        json.loads((checkpoints / f"step-{i:04}.json").read_text())
        for i in range(1, target_steps + 1)
    ]
    if target_steps == maximum:
        write_mesh(out, problem, full, a.solver)
    save(
        out / "summary.json",
        dict(
            completed_steps=target_steps,
            requested_steps=maximum,
            stage=manifest["stage"],
            complete=target_steps == maximum,
            backend="C++ Metal BEM, Mitsuba Metal, PyTorch MPS",
            triangles=len(problem["triangles"]),
            initial_loss=rows[0]["loss"],
            last_preupdate_loss=rows[-1]["loss"],
            heightfield_sha256=digest(out / "hfield.npy"),
        ),
    )


if __name__ == "__main__":
    main()
