"""Capture one upstream CUDA state and compare it with Metal on the same inputs.

Run prepare and metal on the Mac. Run cuda in the authors' Python environment on
a CUDA host with the prepared directory copied there. Results stay in build/.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


UPSTREAM_COMMITS = {
    "0aba879ac1351dd927355db2fc393c0eff7d4438",
    "a7e09916a3189f1d95caead697f867c49bda2cc1",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def load_case(folder):
    case = json.loads((folder / "case.json").read_text())
    for name, expected in case["sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Changed comparison input: {name}")
    pixels = np.load(folder / "hfield.npy", allow_pickle=False)
    problem = json.loads((folder / "problem.json").read_text())
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError("Expected a finite two-dimensional height texture")
    return case, problem, pixels


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.run / "problem.json", args.output / "problem.json")
    shutil.copyfile(args.run / "hfield.npy", args.output / "hfield.npy")
    shutil.copyfile(args.target, args.output / "target.png")
    manifest = json.loads((args.run / "manifest.json").read_text())
    save(
        args.output / "case.json",
        dict(
            config=manifest["config"],
            stage=manifest["stage"],
            frequency_hz=args.frequency,
            sha256={name: digest(args.output / name) for name in ("problem.json", "hfield.npy", "target.png")},
        ),
    )
    load_case(args.output)
    print(f"Prepared {args.output}")


def cuda(args):
    import torch
    import mitsuba as mi

    if not torch.cuda.is_available():
        raise RuntimeError("The upstream capture requires CUDA")
    commit = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit not in UPSTREAM_COMMITS:
        raise ValueError(f"Unrecognized upstream revision: {commit}")
    if subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Expected a clean upstream checkout")
    sys.path.insert(0, str(args.upstream.resolve()))
    mi.set_variant("cuda_ad_rgb")
    import acoustics3d as ac
    from acoustics_opt import acoustic_gradient
    from pyoptim.diffmesh import ImageDiffMesh, rect_points_to_uv
    from pyoptim.losses import ImgImgCLIPLoss

    case, problem, pixels = load_case(args.case)
    config = case["config"]
    bem_options = config["diffbem"]
    points = np.asarray(problem["vertices"], dtype=float)
    faces = np.asarray(problem["triangles"], dtype=int)
    selected = np.asarray(problem["height_vertices"], dtype=int)
    bem = ac.DiffBEM(
        128, 1.5, [case["frequency_hz"]], 1, 1e-5, 1e-5, 1e-5,
        np.asarray(problem["source"]),
        bem_options.get("listener_radius", 50),
        bem_options.get("listener_ds", 5),
        False,
    )
    bem.silent = True
    started = time.monotonic()
    points, faces = bem.precompute(points, faces, selected.tolist())
    if not np.allclose(bem.get_listeners(), problem["listeners"], rtol=0, atol=1e-12):
        raise ValueError("Upstream listeners differ from the prepared input")
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    uv = rect_points_to_uv(points[selected], low[0], high[0], low[2], high[2])
    bem.set_band(case["frequency_hz"])
    acoustic_loss, acoustic_gradient_value = acoustic_gradient(pixels, bem, uv)
    acoustic_seconds = time.monotonic() - started
    model = ImgImgCLIPLoss()
    scene = ImageDiffMesh(np.asarray(points), np.asarray(faces), str(args.case / "target.png"), model)
    resolution = 512 if case["stage"] == 2 else 256
    torch.manual_seed(37)
    torch.cuda.manual_seed_all(37)
    started = time.monotonic()
    view_loss, view_gradient = scene.gradient(
        pixels, np.pi / 2, 0,
        radius=config["image"].get("cam_rad", 1), res=resolution,
    )
    view_seconds = time.monotonic() - started
    output = dict(
        upstream_commit=commit,
        acoustic_module_sha256=digest(next(Path(ac.__file__).parent.glob("*.so"))),
        acoustic_backend="upstream CPU BEM",
        appearance_backend="upstream Mitsuba CUDA and PyTorch CUDA",
        gpu=torch.cuda.get_device_name(),
        input_sha256=digest(args.case / "case.json"),
        acoustic_loss=float(acoustic_loss),
        acoustic_gradient=acoustic_gradient_value.detach().cpu().numpy().tolist(),
        acoustic_seconds=acoustic_seconds,
        view_loss=float(view_loss),
        view_gradient=view_gradient.detach().cpu().numpy().tolist(),
        view_seconds=view_seconds,
    )
    save(args.case / "upstream.json", output)
    print(f"Captured upstream state in {args.case / 'upstream.json'}")


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def metal(args):
    import torch
    from .appearance import Appearance, ImageLoss
    from .native import evaluate

    if not torch.backends.mps.is_available():
        raise RuntimeError("The native comparison requires MPS")
    case, problem, pixels = load_case(args.case)
    upstream = json.loads((args.case / "upstream.json").read_text())
    if upstream["input_sha256"] != digest(args.case / "case.json"):
        raise ValueError("Upstream capture has different inputs")
    started = time.monotonic()
    acoustic = evaluate(
        dict(problem, frequency_hz=case["frequency_hz"]),
        pixels,
        args.case / "metal-input.json",
        args.case / "metal-output.json",
        args.solver,
    )
    acoustic_seconds = time.monotonic() - started
    model = ImageLoss()
    scene = Appearance(problem["vertices"], problem["triangles"], args.case / "target.png", model)
    resolution = 512 if case["stage"] == 2 else 256
    torch.manual_seed(37)
    torch.mps.manual_seed(37)
    started = time.monotonic()
    view_loss, view_gradient = scene.gradient(
        pixels, np.pi / 2, 0,
        radius=case["config"]["image"].get("cam_rad", 1), resolution=resolution,
    )
    view_seconds = time.monotonic() - started
    report = dict(
        upstream_commit=upstream["upstream_commit"],
        input_sha256=digest(args.case / "case.json"),
        metal_solver_sha256=digest(args.solver),
        acoustic_loss_absolute_error=abs(acoustic["loss"] - upstream["acoustic_loss"]),
        acoustic_gradient_relative_error=relative(acoustic["gradient"], upstream["acoustic_gradient"]),
        view_loss_absolute_error=abs(view_loss - upstream["view_loss"]),
        view_gradient_relative_error=relative(view_gradient, upstream["view_gradient"]),
        forward_residual=acoustic["independent_forward_residual"],
        adjoint_residual=acoustic["independent_adjoint_residual"],
        timing=dict(
            upstream_cpu_acoustic_seconds=upstream["acoustic_seconds"],
            metal_acoustic_process_seconds=acoustic_seconds,
            metal_assembly_seconds=acoustic["assembly_seconds"],
            metal_solve_seconds=acoustic["solve_seconds"],
            metal_derivative_seconds=acoustic["derivative_seconds"],
            upstream_cuda_view_seconds=upstream["view_seconds"],
            metal_view_seconds=view_seconds,
        ),
    )
    save(args.case / "comparison.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("run", type=Path)
    p.add_argument("target", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--frequency", type=float, default=1000)
    p.set_defaults(action=prepare)
    p = commands.add_parser("cuda")
    p.add_argument("case", type=Path)
    p.add_argument("upstream", type=Path)
    p.set_defaults(action=cuda)
    p = commands.add_parser("metal")
    p.add_argument("case", type=Path)
    p.add_argument("--solver", type=Path, default=Path("build/BemTextureSolve"))
    p.set_defaults(action=metal)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
