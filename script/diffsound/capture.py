"""Capture independent author-code outputs using CPU substitutions for CUDA placement."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

from . import UPSTREAM


def reference(root):
    modules = {}
    source_hashes = {}

    def load(name, relative, namespace=None, classes=None):
        path = root / relative
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        source = path.read_text().replace(".cuda()", "").replace('"cuda"', '"cpu"').replace("'cuda'", "'cpu'")
        source = source.replace("bempp.api", "bempp_cl.api")
        if classes:
            tree = ast.parse(source)
            source = "\n\n".join(
                ast.get_source_segment(source, n)
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name in classes
            )
        source = source.replace("from torchvision.transforms.functional import gaussian_blur", "")
        module = types.ModuleType(name)
        module.__dict__.update(namespace or {})
        sys.modules[name] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
        modules[name] = module
        return module

    for name in ("src", "src.ddsp", "src.diffelastic"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    scatter = types.ModuleType("torch_scatter")

    def reduce(values, index, dim=0, reduce="sum", dim_size=None):
        shape = list(values.shape)
        shape[dim] = dim_size if dim_size is not None else int(index.max()) + 1
        if reduce == "sum":
            return torch.zeros(shape, dtype=values.dtype).index_add(dim, index, values)
        result = torch.full(shape, torch.iinfo(values.dtype).max, dtype=values.dtype)
        return result.scatter_reduce(dim, index, values, reduce="amin")

    scatter.scatter = reduce
    sys.modules["torch_scatter"] = scatter
    for name in ("material_model", "gauss", "shape_func", "mass_matrix", "mesh", "deform"):
        load("src.diffelastic." + name, "src/diffelastic/" + name + ".py")
    for name in ("utils", "filtered_noise", "oscillator", "mss_loss"):
        load("src.ddsp." + name, "src/ddsp/" + name + ".py")
    load("src.diffelastic.diff_model", "src/diffelastic/diff_model.py")
    shared = dict(torch=torch, np=np, WeightedParam=modules["src.ddsp.oscillator"].WeightedParam)
    for name in ("dmtet_geometry", "dmtet_thickness", "dmtet_interpolate"):
        load(
            name,
            "src/dmtet/geometry/" + name + ".py",
            shared,
            {"DMTet", "PositionalEncoding", "NerfWithPositionEncoding", "DMTetGeometry"},
        )
    import numba

    numba.config.THREADING_LAYER = "workqueue"
    numba.set_num_threads(4)
    load("bem", "src/diffelastic/bem.py")
    return modules, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("external/diffsound"))
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/diffsound/reference.npz"))
    args = parser.parse_args()
    import subprocess

    assert subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip() == UPSTREAM
    torch.manual_seed(23)
    modules, hashes = reference(args.source)
    arrays = {}
    oscillator = modules["src.ddsp.oscillator"]
    force = torch.zeros(2, 17)
    force[:, [0, 2, 6]] = torch.tensor([[0.5, 0.3, 0.1], [0.3, 0.1, 0.4]])
    freq = torch.tensor([853.0, 1907.0, 4913.0], requires_grad=True)
    material = modules["src.diffelastic.material_model"].Material((2700, 7.2e10, 0.19, 6, 1e-7))
    model = oscillator.TraditionalDampedOscillator(force, 2, 3, 2048, 32000, material)
    signal = model(freq)
    arrays.update(
        force=force.numpy(),
        frequencies=freq.detach().numpy(),
        audio=signal.detach().numpy(),
        damped=model.damped_freq.detach().numpy(),
    )
    arrays["audio_gradient"] = torch.autograd.grad(signal.square().mean(), freq)[0].numpy()
    target = model(freq.detach() * 1.037).detach()
    arrays["target"] = target.numpy()
    for kind in ("l1_loss", "geomloss", "rmse_loss"):
        signal = model(freq)
        loss = modules["src.ddsp.mss_loss"].MSSLoss([256, 128], 32000, type=kind)(signal, target, model.damped_freq)
        arrays[kind] = np.array(float(loss.detach()))
        arrays[kind + "_gradient"] = torch.autograd.grad(loss, freq)[0].numpy()
    noise_model = modules["src.ddsp.filtered_noise"].FilteredNoise(2, 512)
    torch.manual_seed(17)
    noise = torch.rand(2, 9, 64) * 2 - 1
    torch.manual_seed(17)
    out = noise_model()
    arrays.update(
        noise_raw=noise_model.coefficient_bank.detach().numpy(),
        noise_input=noise.numpy(),
        noise_output=out.detach().numpy(),
    )
    arrays["noise_gradient"] = torch.autograd.grad(out.square().mean(), noise_model.coefficient_bank)[0].numpy()
    free = oscillator.GTDampedOscillator(force, 2, 3, 512, 32000, [20, 16000], material)
    torch.manual_seed(19)
    free_noise = torch.rand(2, 9, 64) * 2 - 1
    torch.manual_seed(19)
    out = free(noise_rate=0.0005)
    parameters = dict(
        frequency=free.freq_linear.params,
        alpha=free.alpha.params,
        beta=free.beta.params,
        amplitude=free.amp.value,
        noise=free.noise.coefficient_bank,
    )
    gradients = torch.autograd.grad(out.square().mean(), tuple(parameters.values()))
    arrays.update(free_audio=out.detach().numpy(), free_noise_input=free_noise.numpy())
    for (name, p), gradient in zip(parameters.items(), gradients):
        arrays["free_" + name] = p.detach().numpy()
        arrays["free_" + name + "_gradient"] = gradient.numpy()
    curve = oscillator.DampedOscillator(force, 2, 3, 2048, 32000, [20, 16000], material)
    out = curve.forward_curve(freq[:, None], lambda f: 0.002 * f + 4)
    arrays["curve_audio"] = out.detach().numpy()
    loss = modules["src.ddsp.mss_loss"].MSSLoss([256, 128], 32000, type="geomloss")(out, target, curve.damped_freq)
    arrays["curve_loss"] = np.array(float(loss.detach()))
    arrays["curve_gradient"] = torch.autograd.grad(loss, freq)[0].numpy()
    network = modules["dmtet_geometry"].NerfWithPositionEncoding(3, 0.7, 3, 512)
    points = torch.rand(8, 3, requires_grad=True)
    result = network(points)
    parameters = dict(network.named_parameters())
    gradients = torch.autograd.grad(result.square().sum(), (points, *parameters.values()))
    arrays.update(
        network_points=points.detach().numpy(),
        network_output=result.detach().numpy(),
        network_points_gradient=gradients[0].numpy(),
    )
    for (name, p), gradient in zip(parameters.items(), gradients[1:]):
        arrays["network_" + name] = p.detach().numpy()
        arrays["network_" + name + "_gradient"] = gradient.numpy()
    template = modules["dmtet_geometry"].DMTetGeometry.mesh_template_loss
    geometry = types.SimpleNamespace(sdf_nerf=network, grid_res=16)
    distances = -network(points).detach().flatten()
    loss = template(geometry, points, distances, 0)
    arrays.update(
        template_distances=distances.numpy(),
        template_loss=np.array(float(loss.detach())),
        template_gradient=torch.autograd.grad(loss, points)[0].numpy(),
    )
    assert template(geometry, points, -distances, 0) is None
    vertices = (
        torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1.3, 0], [0, 0, 1.7], [1, 1.3, 0], [1, 0, 1.7], [0, 1.3, 1.7], [1, 1.3, 1.7]],
            dtype=torch.float32,
        )
        * 0.1
    )
    tets = torch.tensor([[0, 1, 4, 7], [0, 4, 2, 7], [0, 2, 6, 7], [0, 6, 3, 7], [0, 3, 5, 7], [0, 5, 1, 7]])
    arrays.update(vertices=vertices.numpy(), tets=tets.numpy())
    for order in (1, 2):
        model = modules["src.diffelastic.diff_model"].DiffSoundObj(vertices, tets, mode_num=4, order=order)
        model.eigen_decomposition()
        prefix = f"fem{order}."
        arrays.update(
            {
                prefix + "vertices": model.tetmesh.vertices.numpy(),
                prefix + "mass": model.mass_matrix.to_dense().numpy(),
                prefix + "stiffness": model.stiff_matrix.to_dense().numpy(),
                prefix + "values": model.eigenvalues.numpy(),
            }
        )
    points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    cell = torch.tensor([[0, 1, 2, 3]])
    for case in range(1, 15):
        sdf = torch.tensor([(0.15 + 0.1 * i) * (1 if case & (1 << i) else -1) for i in range(4)], requires_grad=True)
        v, t = modules["dmtet_geometry"].DMTet()(points, sdf, cell)
        arrays[f"clip{case}.sdf"] = sdf.detach().numpy()
        arrays[f"clip{case}.vertices"] = v.detach().numpy()
        arrays[f"clip{case}.tets"] = t.numpy()
        arrays[f"clip{case}.gradient"] = torch.autograd.grad(v.square().sum(), sdf)[0].numpy()
    sdf = torch.tensor([-0.2, 0.1, 0.3, 0.8])
    coef = torch.tensor(0.5, requires_grad=True)
    shell = modules["dmtet_thickness"].DMTet()
    shell.max_thickness = sdf.max()
    _, _, v, t = shell(points, sdf, cell, coef)
    arrays.update(
        shell_sdf=sdf.numpy(),
        shell_vertices=v.detach().numpy(),
        shell_tets=t.numpy(),
        shell_gradient=torch.autograd.grad(v.square().sum(), coef)[0].numpy(),
    )
    vertices = np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.12, 0], [0, 0, 0.15]])
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    p = vertices[faces]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    modes = np.random.default_rng(7).normal(size=(4, 3, 2))
    frequencies = np.array([853.0, 1907.0])
    listener = np.array([[0.5, 0.4, 0.6]])
    model = modules["bem"].BEMModel(vertices, faces)
    transfer = []
    for i, f in enumerate(frequencies):
        neumann = 1.225 * (2 * np.pi * f) ** 2 * (modes[faces, :, i].mean(1) * normals).sum(1)
        model.boundary_equation_solve(neumann, 2 * np.pi * f / 343.0)
        transfer.append(model.potential_solve(listener)[0])
    arrays.update(
        bem_vertices=vertices,
        bem_faces=faces,
        bem_normals=normals,
        bem_modes=modes,
        bem_frequency=frequencies,
        bem_listener=listener,
        bem_transfer=np.array(transfer),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                upstream_commit=UPSTREAM,
                source_hashes=hashes,
                dependencies={
                    p: importlib.metadata.version(p)
                    for p in ("torch", "torchaudio", "numpy", "scipy", "geomloss", "bempp-cl")
                },
                reference_backend="Author PyTorch source on CPU; device placement, unused imports, torch_scatter reductions and bempp.api namespace substituted",
                fixture_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Captured {len(arrays)} arrays to {args.output}")


if __name__ == "__main__":
    main()
