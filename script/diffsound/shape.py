"""Released eigenvalue-supervised thickness, morphing, and neural shape inference."""

import torch

from . import DEVICE
from .dsp import Weighted
from .fem import Elasticity, MATERIALS, read_mesh, compact
from .geometry import NeuralSDF, clip, largest_component, load_grid, save_mesh, signed_distance, surface, voxel_samples
from .runtime import train, save_result


def elastic_mesh(vertices, tets):
    vertices, tets = largest_component(vertices, tets)
    p = vertices[tets]
    volume = torch.linalg.det((p[:, :3] - p[:, 3:]).transpose(1, 2)).abs()
    return compact(vertices, tets[volume > 0])


def eigenvalues(vertices, tets, material, modes):
    vertices, tets = elastic_mesh(vertices, tets)
    fem = Elasticity(vertices, tets)
    pairs = fem.solve(material[1], material[2], material[0], modes)
    return fem.values(material[1], material[2], material[0], pairs)


def generate(root, output, config, kind):
    points, tets = load_grid(root, config["generation_grid"], config["scale"])
    first = signed_distance(root / f"mesh/{config['first']}.obj", points)
    second = signed_distance(root / f"mesh/{config['second']}.obj", points) if kind == "morphing" else None
    for target in config["targets"]:
        if (output / f"{kind}-{target:g}.obj").exists():
            continue
        if kind == "thickness":
            mesh = clip(points, first, tets, target * first.max())
        else:
            mesh = clip(points, target * first + (1 - target) * second, tets)
        save_mesh(output / f"{kind}-{target:g}.obj", mesh[2], mesh[3])


def scalar(root, output, config, kind):
    targets = output / "targets"
    targets.mkdir(exist_ok=True)
    generate(root, targets, config, kind)
    points, tets = load_grid(root, config["grid"], config["scale"])
    first = signed_distance(root / f"mesh/{config['first']}.obj", points)
    second = signed_distance(root / f"mesh/{config['second']}.obj", points) if kind == "morphing" else None
    material = MATERIALS[config["material"]]
    for index, target in enumerate(config["targets"]):
        directory = output / f"{kind}-{target:g}"
        if (directory / "result.json").exists():
            continue
        torch.manual_seed(config["seed"] + index + 1)
        directory.mkdir(exist_ok=True)
        sdf = signed_distance(targets / f"{kind}-{target:g}.obj", points)
        with torch.no_grad():
            mesh = clip(points, sdf, tets)
            truth = eigenvalues(mesh[0], mesh[1], material, config["modes"])
        model = Weighted(torch.linspace(0, 1, 32), device="cpu")

        def mesh():
            coefficient = model()
            if kind == "thickness":
                return clip(points, first, tets, coefficient * first.max())
            return clip(points, coefficient * first + (1 - coefficient) * second, tets)

        def closure(i, cache):
            v, t, _, _ = mesh()
            vals = eigenvalues(v, t, material, config["modes"])
            return ((vals - truth) / truth).square().mean(), dict(coefficient=float(model().detach()))

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        state = train(directory, "fit", model, config["steps"], closure, optimizer)
        with torch.no_grad():
            v, t, _, _ = mesh()
            v, t = elastic_mesh(v, t)
            faces, _, _ = surface(v, t)
            vals = eigenvalues(v, t, material, config["modes"])
        save_mesh(directory / "result.obj", v, torch.from_numpy(faces))
        save_result(
            directory,
            dict(
                target=target,
                coefficient=float(model().detach()),
                coefficient_absolute_error=abs(float(model().detach()) - target),
                relative_eigenvalue_error=float(
                    torch.linalg.vector_norm(vals - truth) / torch.linalg.vector_norm(truth)
                ),
                relative_eigenvalue_mse=float(((vals - truth) / truth).square().mean()),
                steps=state["step"],
                complete_paper_reproduction=False,
                supervision="Target eigenvalues",
            ),
            vertices=v,
            tets=t,
            values=vals,
            target_values=truth,
        )


def reconstruct(root, output, config):
    for index, name in enumerate(config["shapes"]):
        directory = output / name
        if (directory / "result.json").exists():
            continue
        torch.manual_seed(config["seed"] + index + 1)
        torch.mps.manual_seed(config["seed"] + index + 1)
        directory.mkdir(exist_ok=True)
        vertices, tets = read_mesh(root / f"mesh/shape/{name}.msh")
        with torch.no_grad():
            truth = eigenvalues(vertices, tets, MATERIALS["Ceramic"], config["modes"])
        queries, distances, scale = voxel_samples(root / f"mesh/shape/{name}_surf.obj", config["voxels"])
        points, grid = load_grid(root, config["grid"], scale)
        model = NeuralSDF(points, grid, config["grid"], scale, config["frequencies"])

        def template(i, cache):
            return model.template_loss(queries, distances), {}

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        train(directory, "pretrain", model, config["pretrain_steps"], template, optimizer)

        def closure(i, cache):
            v, t, _, _ = model()
            vals = eigenvalues(v, t, MATERIALS["Ceramic"], config["modes"])
            audio_loss = ((vals - truth) / truth).square().mean().sqrt()
            constraint = model.template_loss(queries, distances)
            if constraint is None:
                constraint = torch.zeros((), device=DEVICE)
            loss = constraint + 0.0002 * audio_loss.float().to(DEVICE)
            if float(loss.detach()) < cache.get("best_loss", float("inf")):
                cache["best_loss"] = float(loss.detach())
                cache["best_step"] = i
                torch.save(model.state_dict(), directory / "best.pt")
            return loss, dict(
                relative_eigenvalue_rmse=float(audio_loss.detach()), template_loss=float(constraint.detach())
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.8)
        state = train(directory, "fit", model, config["steps"], closure, optimizer, scheduler)
        model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=True))
        with torch.no_grad():
            v, t, _, _ = model()
            v, t = elastic_mesh(v, t)
            faces, _, _ = surface(v, t)
            vals = eigenvalues(v, t, MATERIALS["Ceramic"], config["modes"])
        save_mesh(directory / "result.obj", v, torch.from_numpy(faces))
        save_mesh(directory / "result.msh", v, t)
        save_result(
            directory,
            dict(
                steps=state["step"],
                best_step=state["extra"]["best_step"],
                relative_eigenvalue_error=float(
                    torch.linalg.vector_norm(vals - truth) / torch.linalg.vector_norm(truth)
                ),
                relative_eigenvalue_rmse=float(((vals - truth) / truth).square().mean().sqrt()),
                complete_paper_reproduction=False,
                supervision="Coarse occupancy constraints and target eigenvalues",
            ),
            vertices=v,
            tets=t,
            values=vals,
            target_values=truth,
        )
