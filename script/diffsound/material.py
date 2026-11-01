"""Synthetic and recorded-audio material inference with the released schedules."""

import math
import numpy as np
from scipy.interpolate import interp1d
import soundfile as sf
import torch
import torchaudio.functional as AF
import yaml

from . import DEVICE
from .dsp import ModalFit, SpectralLoss, modified_sigmoid, render
from .fem import Elasticity, MATERIALS, Material, read_mesh
from .runtime import train, save_result


def recorded_audio(root, count=8, rate=32000, samples=8000):
    metadata = yaml.safe_load((root / "real_audio/metadata.yaml").read_text())
    audio = []
    for path in sorted((root / "real_audio").glob("*mic*.wav"))[:count]:
        waveform, sr = sf.read(path, dtype="float32", always_2d=True)
        x = torch.tensor(waveform.T[:, metadata["pad"][1] * sr :])
        x = AF.gain(x, metadata["gain"][1])
        x = AF.resample(x, sr, rate)[0, :samples]
        x = AF.highpass_biquad(x, rate, 100)
        if len(x) != samples or x.abs().max() == 0:
            raise ValueError(f"Invalid recording {path}")
        audio.append(x / x.abs().max())
    if len(audio) != count:
        raise ValueError("Missing released recordings")
    return torch.stack(audio).to(DEVICE)


def fit_curve(output, target, material, config):
    model = ModalFit(len(target), config["modes"] * 32, config["samples"], material)
    force = torch.zeros(len(target), config["force_samples"], device=DEVICE)
    force[:, 0] = 1
    loss_fn = SpectralLoss([512, 256, 128, 64, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.99)
    train(
        output,
        "damping",
        model,
        config["damping_steps"],
        lambda i, cache: (loss_fn(model(force, config["rate"]), target), {}),
        optimizer,
        scheduler,
    )
    with torch.no_grad():
        f, d = model.controls()
    f, d = f.flatten().cpu().numpy(), d.flatten().cpu().numpy()
    x, y = [], []
    for low in range(20, 20000, 500):
        selected = (f > low) & (f < low + 500) & (d < 100)
        if selected.any():
            x.append(low + 250)
            y.append(float(d[selected].min()))
    if len(x) < 2:
        raise RuntimeError("The released damping selection yielded fewer than two interpolation points")
    np.savez(output / "damping_curve.npz", frequency=x, damping=y)
    return interp1d(x, y, fill_value="extrapolate")


def fit_material(output, vertices, tets, initial, target, config, curve=None, fit_gains=False):
    fem = Elasticity(vertices, tets, 1 if config["exp_mode"] in (0, 2) else 2)
    model = Material(initial, config["exp_mode"] in (2, 3))
    if not (output / "material.pt").exists():
        model.initialize(config["material_init_steps"])
    if fit_gains:
        model.register_parameter(
            "amplitude",
            torch.nn.Parameter(torch.empty(len(target), config["modes"], 1, device=DEVICE).uniform_(0, 0.04)),
        )
    force = torch.zeros(len(target), config["force_samples"], device=DEVICE)
    force[:, 0] = 1
    early = SpectralLoss([2048, 1024], "transport", config["rate"])
    late = SpectralLoss([1024, 512, 256, 128, 64], "l1", config["rate"])
    rmse = SpectralLoss([1024, 512, 256, 128, 64], "rmse", config["rate"])

    def sound(cache, refresh=False):
        e, nu = model.youngs(), model.poisson()
        if refresh or not cache:
            pairs = fem.solve(e.detach().double(), nu.detach().double(), initial[0], config["modes"])
            cache.update(values=pairs[0], vectors=pairs[1], projected=fem.projected(pairs))
        kl, km, mass = cache["projected"]
        lam = e.double() * nu.double() / ((1 + nu.double()) * (1 - 2 * nu.double()))
        mu = e.double() / (2 * (1 + nu.double()))
        values = cache["values"] + lam * kl + mu * km - cache["values"] * initial[0] * mass
        f = (values.sqrt() / (2 * math.pi)).float().to(DEVICE)
        if curve is None:
            d = 0.5 * (initial[3] + initial[4] * (2 * math.pi * f).square())
        else:
            # The release freezes the damping interpolator's frequency derivative.
            d = torch.tensor(curve(f.detach().cpu().numpy()), dtype=torch.float32, device=DEVICE)
        gains = (
            modified_sigmoid(model.amplitude)
            if fit_gains
            else torch.ones(len(target), config["modes"], 1, device=DEVICE)
        )
        audio, damped = render(f, d, gains, force, config["samples"], config["rate"])
        if curve is not None:
            audio = audio / audio.abs().amax(1, keepdim=True)
            damped = damped[:1, :, :1]
        return audio, f, d, damped

    def closure(i, cache):
        audio, _, _, damped = sound(cache, i % 15 == 0)
        loss = (early if i < config["early_steps"] else late)(audio, target, damped)
        return loss, dict(youngs_pa=float(model.youngs().detach()), poisson=float(model.poisson().detach()))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 if curve is not None else 0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.9)

    def reset():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.95)

    state = train(
        output, "material", model, config["steps"], closure, optimizer, scheduler, (config["early_steps"], reset)
    )
    with torch.no_grad():
        audio, f, d, _ = sound(state["extra"], refresh=True)
        error = float(rmse(audio, target))
    sf.write(output / "prediction.wav", audio[0].cpu().numpy(), config["rate"], subtype="FLOAT")
    sf.write(output / "target.wav", target[0].cpu().numpy(), config["rate"], subtype="FLOAT")
    save_result(
        output,
        dict(
            steps=state["step"],
            youngs_pa=float(model.youngs().detach()),
            poisson=float(model.poisson().detach()),
            density=initial[0],
            alpha=initial[3],
            beta=initial[4],
            order=fem.order,
            rate=config["rate"],
            force_samples=config["force_samples"],
            youngs_over_density=float(model.youngs().detach()) / initial[0],
            rmse=error,
            complete_paper_reproduction=False,
            contact="Known impulse",
            elapsed_seconds=state["elapsed_seconds"],
        ),
        audio=audio,
        target=target,
        frequency=f,
        damping=d,
        vertices=vertices,
        tets=tets,
        gains=modified_sigmoid(model.amplitude) if fit_gains else torch.ones(len(target), config["modes"], 1),
    )


def run(root, output, config, real=False):
    vertices, tets = read_mesh(root / "mesh/bowl/bowl.obj")
    if real:
        target = recorded_audio(root, config["recordings"], config["rate"], config["samples"])
        material = MATERIALS[config["material"]]
        curve = fit_curve(output, target, material, config)
        fit_material(output, vertices, tets, material, target, config, curve)
        return
    random_path = output / "materials.npz"
    if not random_path.exists():
        init_e = torch.rand(16) * 9e10 + 1e10
        init_n = torch.rand(16) * 0.3 + 0.1
        target_e = torch.rand(16) * 9e10 + 1e10
        target_n = torch.rand(16) * 0.3 + 0.1
        np.savez(
            random_path,
            init_e=init_e.numpy(),
            init_n=init_n.numpy(),
            target_e=target_e.numpy(),
            target_n=target_n.numpy(),
        )
    with np.load(random_path) as z:
        materials = dict(z)
    truth = Elasticity(vertices, tets, 2)
    for case in range(config["cases"]):
        directory = output / f"case-{case:02d}"
        if (directory / "result.json").exists():
            continue
        torch.manual_seed(config["seed"] + case + 1)
        torch.mps.manual_seed(config["seed"] + case + 1)
        directory.mkdir(exist_ok=True)
        initial = (2700, float(materials["init_e"][case]), float(materials["init_n"][case]), 6, 1e-7)
        gt = (2700, float(materials["target_e"][case]), float(materials["target_n"][case]), 6, 1e-7)
        with torch.no_grad():
            vals, _ = truth.solve(gt[1], gt[2], gt[0], config["modes"])
            f = (vals.sqrt() / (2 * math.pi)).float().to(DEVICE)
            force = torch.zeros(1, config["force_samples"], device=DEVICE)
            force[:, 0] = 1
            target, _ = render(
                f,
                0.5 * (gt[3] + gt[4] * (2 * math.pi * f).square()),
                torch.ones(1, config["modes"], 1, device=DEVICE),
                force,
                config["samples"],
                config["rate"],
            )
        np.savez(directory / "ground_truth.npz", material=gt)
        fit_material(directory, vertices, tets, initial, target, config)
