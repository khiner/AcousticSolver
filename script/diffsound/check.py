"""Compare native outputs/gradients with retained author fixtures and finite differences."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial import cKDTree
import torch

from . import DEVICE
from .dsp import FilteredNoise, ModalFit, SpectralLoss, render
from .fem import Elasticity
from .geometry import NeuralSDF, clip
from .impact import acoustic_transfer, candidate_gains, similarities
from .runtime import train


def restart_checks(equal):
    with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
        root = Path(temporary)

        def run(path, steps):
            torch.manual_seed(12)
            torch.mps.manual_seed(12)
            model = torch.nn.ParameterDict(
                {"host": torch.nn.Parameter(torch.ones(3)), "metal": torch.nn.Parameter(torch.ones(3, device=DEVICE))}
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.8)

            def reset():
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.9)

            def closure(i, cache):
                if i % 2 == 0:
                    cache["target"] = torch.rand(3)
                target = cache["target"].to(DEVICE) + torch.rand(3, device=DEVICE) * 0.1
                return ((model["host"].to(DEVICE) * model["metal"] - target) ** 2).sum(), {}

            state = train(path, "fit", model, steps, closure, optimizer, scheduler, (4, reset))
            return state, torch.rand(3), torch.rand(3, device=DEVICE)

        expected = run(root / "continuous", 7)
        for split in (3, 5):
            run(root / str(split), split)
            actual = run(root / str(split), 7)
            for key in ("host", "metal"):
                equal(
                    f"restart {split} {key}",
                    actual[0]["model"][key],
                    expected[0]["model"][key].cpu().numpy(),
                    rtol=0,
                    atol=0,
                )
            for i in (1, 2):
                equal(f"restart {split} RNG {i}", actual[i], expected[i].cpu().numpy(), rtol=0, atol=0)
            equal(
                f"restart {split} cache",
                actual[0]["extra"]["target"],
                expected[0]["extra"]["target"].numpy(),
                rtol=0,
                atol=0,
            )
        model = torch.nn.Linear(1, 1)

        def closure(i, cache):
            return (model.weight.square().sum() if i < 2 else None), {}

        for _ in range(2):
            state = train(root / "converged", "fit", model, 10, closure, torch.optim.Adam(model.parameters()))
            equal("converged stage remains stopped", np.array(state["step"]), np.array(2), rtol=0, atol=0)
            assert state["converged"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("tests/fixtures/diffsound/reference.npz"))
    args = parser.parse_args()
    metadata = json.loads(args.fixture.with_suffix(".json").read_text())
    assert hashlib.sha256(args.fixture.read_bytes()).hexdigest() == metadata["fixture_sha256"]
    checks = []

    def equal(name, a, b, rtol=2e-4, atol=2e-5):
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=name)
        checks.append(name)

    with np.load(args.fixture) as z:
        force = torch.tensor(z["force"], device=DEVICE)
        frequency = torch.tensor(z["frequencies"], device=DEVICE, requires_grad=True)

        def signal():
            d = 0.5 * (6 + 1e-7 * (2 * torch.pi * frequency).square())
            return render(frequency, d, torch.ones(1, 3, 1, device=DEVICE), force, 2048)

        audio, damped = signal()
        equal("audio", audio, z["audio"], atol=1.2e-4)
        equal("audio gradient", torch.autograd.grad(audio.square().mean(), frequency)[0], z["audio_gradient"])
        target = torch.tensor(z["target"], device=DEVICE)
        for kind, reference in (("l1", "l1_loss"), ("transport", "geomloss"), ("rmse", "rmse_loss")):
            audio, damped = signal()
            loss = SpectralLoss([256, 128], kind)(audio, target, damped)
            equal(kind + " loss", loss, z[reference], rtol=5e-4)
            equal(
                kind + " gradient",
                torch.autograd.grad(loss, frequency)[0],
                z[reference + "_gradient"],
                rtol=0.01 if kind == "rmse" else 0.005,
                atol=2e-5,
            )
        noise = FilteredNoise(2, 512)
        with torch.no_grad():
            noise.raw.copy_(torch.tensor(z["noise_raw"], device=DEVICE))
        out = noise(torch.tensor(z["noise_input"], device=DEVICE))
        equal("noise", out, z["noise_output"])
        equal("noise gradient", torch.autograd.grad(out.square().mean(), noise.raw)[0], z["noise_gradient"], atol=2e-8)
        free = ModalFit(2, 3, 512, (2700, 7.2e10, 0.19, 6, 1e-7))
        parameters = dict(
            frequency=free.frequency.raw,
            alpha=free.alpha.raw,
            beta=free.beta.raw,
            amplitude=free.amplitude,
            noise=free.noise.raw,
        )
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(torch.tensor(z["free_" + name], device=DEVICE))
        out = free(force, noise=torch.tensor(z["free_noise_input"], device=DEVICE))
        equal("free modal audio", out, z["free_audio"], atol=1.2e-4)
        gradients = torch.autograd.grad(out.square().mean(), tuple(parameters.values()))
        for name, gradient in zip(parameters, gradients):
            equal("free modal " + name + " gradient", gradient, z["free_" + name + "_gradient"], rtol=0.001, atol=2e-6)
        out, damped = render(frequency, 0.002 * frequency.detach() + 4, torch.ones(2, 3, 1, device=DEVICE), force, 2048)
        out = out / out.abs().amax(1, keepdim=True)
        equal("curve audio", out, z["curve_audio"], atol=1.2e-4)
        loss = SpectralLoss([256, 128], "transport")(out, target, damped[:1, :, :1])
        equal("curve transport", loss, z["curve_loss"], rtol=5e-4)
        equal(
            "curve transport gradient",
            torch.autograd.grad(loss, frequency)[0],
            z["curve_gradient"],
            rtol=0.005,
            atol=2e-5,
        )
        network = NeuralSDF(torch.zeros(4, 3), torch.tensor([[0, 1, 2, 3]]), 16, 0.7)
        names = ["layer_0", "layers.0", "layers.1", "layers.2", "final_layer"]
        parameters = {}
        with torch.no_grad():
            for i, name in enumerate(names):
                for field in ("weight", "bias"):
                    p = getattr(network.network[2 * i], field)
                    key = name + "." + field
                    p.copy_(torch.tensor(z["network_" + key], device=DEVICE))
                    parameters[key] = p
        points = torch.tensor(z["network_points"], device=DEVICE, requires_grad=True)
        out = network.sdf(points)
        equal("SDF output", out, z["network_output"].flatten())
        gradients = torch.autograd.grad(out.square().sum(), (points, *parameters.values()))
        equal("SDF input gradient", gradients[0], z["network_points_gradient"])
        for name, gradient in zip(parameters, gradients[1:]):
            equal("SDF " + name + " gradient", gradient, z["network_" + name + "_gradient"])
        distances = torch.tensor(z["template_distances"], device=DEVICE)
        loss = network.template_loss(points, distances)
        equal("occupancy loss", loss, z["template_loss"])
        equal("occupancy gradient", torch.autograd.grad(loss, points)[0], z["template_gradient"])
        assert network.template_loss(points, -distances) is None
        for order in (1, 2):
            vertices = torch.tensor(z["vertices"], dtype=torch.float64)
            tets = torch.tensor(z["tets"])
            fem = Elasticity(vertices, tets, order)
            prefix = f"fem{order}."
            distance, ids = cKDTree(z[prefix + "vertices"]).query(fem.vertices.numpy())
            assert distance.max() < 1e-7 and len(np.unique(ids)) == len(ids)
            dofs = (ids[:, None] * 3 + np.arange(3)).reshape(-1)
            k = fem.sparse(fem.stiffness(7.2e10, 0.19)).toarray()
            mass = fem.sparse(fem.mass * 2700).toarray()
            for label, actual, expected in (
                ("mass", mass, z[prefix + "mass"][np.ix_(dofs, dofs)]),
                ("stiffness", k, z[prefix + "stiffness"][np.ix_(dofs, dofs)]),
            ):
                equal(prefix + label, actual, expected, atol=2e-7 * abs(expected).max())
            pairs = fem.solve(7.2e10, 0.19, 2700, 4)
            equal(prefix + "eigenvalues", pairs[0], z[prefix + "values"], rtol=3e-5, atol=1e-5)
            youngs = torch.tensor(7.2e10, dtype=torch.float64, requires_grad=True)
            poisson = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
            weights = torch.tensor([1.0, 0.7, 0.2, 0.1], dtype=torch.float64) / pairs[0]
            loss = (fem.values(youngs, poisson, 2700, pairs) * weights).sum()
            dy, dn = torch.autograd.grad(loss, (youngs, poisson))
            delta = 1e-4
            minus = fem.solve(7.2e10, 0.19 - delta, 2700, 4)[0]
            plus = fem.solve(7.2e10, 0.19 + delta, 2700, 4)[0]
            equal(prefix + "Young gradient", dy, 2 / 7.2e10, rtol=1e-6, atol=1e-18)
            equal(prefix + "Poisson gradient", dn, float(((plus - minus) * weights).sum() / (2 * delta)), rtol=1e-4)
            v = vertices.clone().requires_grad_()
            shape = Elasticity(v, tets, order)
            value = (shape.values(7.2e10, 0.19, 2700, pairs) * weights).sum()
            gradient = torch.autograd.grad(value, v)[0]
            equal(prefix + "scale derivative", (gradient * v).sum(), -4.0, rtol=1e-5)
        points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        cell = torch.tensor([[0, 1, 2, 3]])
        for case in range(1, 15):
            sdf = torch.tensor(z[f"clip{case}.sdf"], requires_grad=True)
            v, t, _, _ = clip(points, sdf, cell)
            equal(f"clip {case} vertices", v, z[f"clip{case}.vertices"])
            equal(f"clip {case} topology", t, z[f"clip{case}.tets"], rtol=0, atol=0)
            equal(f"clip {case} gradient", torch.autograd.grad(v.square().sum(), sdf)[0], z[f"clip{case}.gradient"])
        thickness = torch.tensor(0.5, requires_grad=True)
        sdf = torch.tensor(z["shell_sdf"])
        v, t, _, _ = clip(points, sdf, cell, thickness * sdf.max())
        equal("shell vertices", v, z["shell_vertices"])
        equal("shell topology", t, z["shell_tets"], rtol=0, atol=0)
        equal("shell gradient", torch.autograd.grad(v.square().sum(), thickness)[0], z["shell_gradient"])
        transfer = acoustic_transfer(
            z["bem_vertices"], z["bem_faces"], z["bem_normals"], z["bem_modes"], z["bem_frequency"], z["bem_listener"]
        )
        equal("BEM pressure", transfer, z["bem_transfer"])
        indices, gains = candidate_gains(
            z["bem_vertices"],
            z["bem_faces"],
            z["bem_normals"],
            np.ones(4),
            z["bem_modes"],
            z["bem_frequency"],
            transfer,
        )
        scores = similarities(gains, gains)
        equal("impact self similarity", scores.diagonal(), np.ones(len(indices)))
    restart_checks(equal)
    print(json.dumps(dict(passed=len(checks), checks=checks), indent=2))


if __name__ == "__main__":
    main()
