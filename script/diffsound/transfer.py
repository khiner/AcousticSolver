"""Transfer inferred material to another mesh and fit its modal gains and noise."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import soundfile as sf
import torch
from torch import nn
import torchaudio.functional as AF

from . import DEVICE
from .dsp import FilteredNoise, SpectralLoss, modified_sigmoid, render
from .fem import Elasticity, read_mesh
from .runtime import prepare, train, save_result


class ModalGainsAndNoise(nn.Module):
    def __init__(self, recordings, modes, samples):
        super().__init__()
        self.amplitude = nn.Parameter(torch.empty(recordings, modes, 1, device=DEVICE).uniform_(0, 0.04))
        self.noise = FilteredNoise(recordings, samples)

    def forward(self, frequency, damping, force, rate):
        audio, _ = render(frequency, damping, modified_sigmoid(self.amplitude), force, self.noise.samples, rate)
        return audio + 0.0005 * self.noise()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-run", type=Path, required=True)
    parser.add_argument(
        "--mesh", type=Path, required=True, help="Tetrahedral mesh in the same length units as the source object"
    )
    parser.add_argument(
        "--audio", type=Path, nargs="+", required=True, help="Onset-aligned recordings of the destination object"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = json.loads((args.material_run / "result.json").read_text())
    with np.load(args.material_run / "result.npz") as z:
        samples = z["target"].shape[-1]
        modes = len(z["frequency"])
    config = dict(
        steps=args.steps,
        samples=samples,
        modes=modes,
        rate=result["rate"],
        force_samples=result["force_samples"],
        learning_rate=0.005,
        noise_gain=0.0005,
        reconstruction="Figure 8 material transfer: fixed material and damping, fit modal amplitudes and noise; author driver and plate assets are absent",
    )
    sources = [args.material_run / "result.json", args.material_run / "result.npz", args.mesh, *args.audio]
    if args.mesh.suffix == ".obj":
        sources.append(Path(str(args.mesh) + "_.msh"))
    curve_path = args.material_run / "damping_curve.npz"
    if curve_path.exists():
        sources.append(curve_path)
    prepare(args.output, config, sources, args.resume)
    torch.manual_seed(0)
    torch.mps.manual_seed(0)
    vertices, tets = read_mesh(args.mesh)
    values, _ = Elasticity(vertices, tets, result["order"]).solve(
        result["youngs_pa"], result["poisson"], result["density"], modes
    )
    frequency = (values.sqrt() / (2 * torch.pi)).float().to(DEVICE)
    damping = 0.5 * (result["alpha"] + result["beta"] * (2 * torch.pi * frequency).square())
    if curve_path.exists():
        with np.load(curve_path) as z:
            curve = interp1d(z["frequency"], z["damping"], fill_value="extrapolate")
        damping = torch.tensor(curve(frequency.cpu().numpy()), dtype=torch.float32, device=DEVICE)
    target = []
    for path in args.audio:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        x = AF.resample(torch.tensor(audio.mean(1)), rate, config["rate"])[:samples]
        if len(x) != samples or x.abs().max() == 0:
            raise ValueError(f"Insufficient or silent audio: {path}")
        target.append(x / x.abs().max())
    target = torch.stack(target).to(DEVICE)
    force = torch.zeros(len(target), config["force_samples"], device=DEVICE)
    force[:, 0] = 1
    model = ModalGainsAndNoise(len(target), modes, samples)
    loss = SpectralLoss([1024, 512, 256, 128, 64], rate=config["rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.99)
    state = train(
        args.output,
        "appearance",
        model,
        args.steps,
        lambda i, c: (loss(model(frequency, damping, force, config["rate"]), target), {}),
        optimizer,
        scheduler,
    )
    with torch.no_grad():
        audio = model(frequency, damping, force, config["rate"])
    for i, (a, b) in enumerate(zip(audio.cpu().numpy(), target.cpu().numpy())):
        sf.write(args.output / f"prediction-{i}.wav", a, config["rate"], subtype="FLOAT")
        sf.write(args.output / f"target-{i}.wav", b, config["rate"], subtype="FLOAT")
    save_result(
        args.output,
        dict(
            steps=state["step"],
            youngs_pa=result["youngs_pa"],
            poisson=result["poisson"],
            rmse=float(SpectralLoss([1024, 512, 256, 128, 64], "rmse")(audio, target)),
            complete_paper_reproduction=False,
            reconstruction=config["reconstruction"],
        ),
        audio=audio,
        target=target,
        frequency=frequency,
        damping=damping,
        gains=modified_sigmoid(model.amplitude),
        vertices=vertices,
        tets=tets,
    )


if __name__ == "__main__":
    main()
