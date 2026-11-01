"""Section 4.4 impact localization using modal gains and acoustic transfer."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import torch

from . import DEVICE
from .fem import Elasticity
from .geometry import surface
from .material import fit_material
from .runtime import prepare, save_result


def acoustic_transfer(vertices, faces, normal, modes, frequencies, listener):
    """The released exterior Helmholtz boundary equation with a Numba CPU backend."""
    import numba

    numba.config.THREADING_LAYER = "workqueue"
    numba.set_num_threads(4)
    import bempp_cl.api as bem

    grid = bem.Grid(vertices.T.astype(np.float64), faces.T.astype(np.uint32))
    space = bem.function_space(grid, "DP", 0)
    settings = dict(precision="single", device_interface="numba")
    identity = bem.operators.boundary.sparse.identity(space, space, space, **settings)
    response = []
    for j, f in enumerate(frequencies):
        omega = 2 * np.pi * f
        k = omega / 343.0
        # Normal pressure derivative for exp(-i*omega*t) and unit modal displacement.
        displacement = modes[:, :, j][faces].mean(1)
        neumann = bem.GridFunction(space, coefficients=1.225 * omega**2 * (displacement * normal).sum(1))
        double = bem.operators.boundary.helmholtz.double_layer(space, space, space, k, **settings)
        single = bem.operators.boundary.helmholtz.single_layer(space, space, space, k, **settings)
        pressure, info, residuals = bem.linalg.gmres(
            -0.5 * identity + double, single * neumann, tol=1e-6, return_residuals=True
        )
        if info != 0:
            raise RuntimeError(f"BEM solve did not converge at {f:g} Hz: {info}")
        points = np.asarray(listener, dtype=np.float64).reshape(3, 1)
        s = bem.operators.potential.helmholtz.single_layer(space, points, k, **settings)
        d = bem.operators.potential.helmholtz.double_layer(space, points, k, **settings)
        response.append(complex((-s * neumann + d * pressure).reshape(-1)[0]))
    return np.asarray(response)


def candidate_gains(vertices, faces, normals, areas, modes, frequencies, transfer):
    vertex_normal = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normal, faces[:, i], normals * areas[:, None])
    candidates = np.unique(faces)
    n = vertex_normal[candidates]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    excitation = np.einsum("vcm,vc->vm", modes[candidates], n)
    # Generalized eigenvectors are mass-normalized; a unit impulse contributes 1/omega.
    return candidates, np.abs(excitation * transfer[None, :] / (2 * np.pi * frequencies[None, :]))


def similarities(observed, candidates):
    a = observed / np.maximum(np.linalg.norm(observed, axis=-1, keepdims=True), 1e-30)
    b = candidates / np.maximum(np.linalg.norm(candidates, axis=-1, keepdims=True), 1e-30)
    return a @ b.T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-run", type=Path, required=True)
    parser.add_argument(
        "--listener", nargs=3, type=float, required=True, help="Microphone position in mesh coordinates"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--early-steps", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = json.loads((args.material_run / "result.json").read_text())
    with np.load(args.material_run / "result.npz") as z:
        vertices = torch.tensor(z["vertices"], dtype=torch.float64)
        tets = torch.tensor(z["tets"])
        target = torch.tensor(z["target"], device=DEVICE)
        modes = len(z["frequency"])
    config = dict(
        steps=args.steps,
        early_steps=args.early_steps,
        listener=args.listener,
        material_run=str(args.material_run.resolve()),
        reconstruction="Paper Section 4.4 plus released BEM helper; no upstream localization driver or microphone calibration is supplied",
        density=result["density"],
        rate=result["rate"],
        force_samples=result["force_samples"],
        order=result["order"],
        modes=modes,
        samples=target.shape[-1],
        exp_mode=2 if result["order"] == 1 else 3,
        material_init_steps=5000,
    )
    sources = [args.material_run / "result.npz", args.material_run / "result.json"]
    curve_path = args.material_run / "damping_curve.npz"
    curve = None
    if curve_path.exists():
        sources.append(curve_path)
        with np.load(curve_path) as z:
            curve = interp1d(z["frequency"], z["damping"], fill_value="extrapolate")
    prepare(args.output, config, sources, args.resume)
    torch.manual_seed(0)
    torch.mps.manual_seed(0)
    directory = args.output / "joint-fit"
    directory.mkdir(exist_ok=True)
    if not (directory / "result.json").exists():
        initial = (result["density"], result["youngs_pa"], result["poisson"], result["alpha"], result["beta"])
        fit_material(directory, vertices, tets, initial, target, config, curve, fit_gains=True)
    result = json.loads((directory / "result.json").read_text())
    with np.load(directory / "result.npz") as z:
        gains = z["gains"][..., 0]
    fem = Elasticity(vertices, tets, config["order"])
    values, vectors = fem.solve(result["youngs_pa"], result["poisson"], config["density"], modes)
    frequency = values.sqrt().numpy() / (2 * np.pi)
    modes = vectors.numpy().reshape(-1, 3, modes)[: len(vertices)]
    faces, normals, areas = surface(vertices, tets)
    transfer_path = args.output / "transfer.npz"
    if transfer_path.exists():
        with np.load(transfer_path) as z:
            transfer = z["transfer"]
    else:
        transfer = acoustic_transfer(vertices.numpy(), faces, normals, modes, frequency, args.listener)
        np.savez(transfer_path, transfer=transfer)
    candidates, predicted = candidate_gains(vertices.numpy(), faces, normals, areas, modes, frequency, transfer)
    scores = similarities(gains, predicted)
    selected = candidates[scores.argmax(1)]
    save_result(
        args.output,
        dict(
            steps=result["steps"],
            vertices=selected.tolist(),
            positions=vertices[selected].tolist(),
            complete_paper_reproduction=False,
            reconstruction=config["reconstruction"],
        ),
        observed_gains=gains,
        candidate_gains=predicted,
        candidates=candidates,
        similarity=scores,
        transfer=transfer,
    )


if __name__ == "__main__":
    main()
