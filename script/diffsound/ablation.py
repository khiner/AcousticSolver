"""Section 4.1 frequency-scaling loss comparison on the released bowl mesh."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import DEVICE
from .dsp import SpectralLoss, render
from .fem import Elasticity, MATERIALS, read_mesh
from .runtime import prepare, train, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/diffsound/source/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--early-steps", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = dict(
        steps=args.steps,
        early_steps=args.early_steps,
        rate=32000,
        samples=8000,
        modes=16,
        learning_rate=0.005,
        reconstruction="Section 4.1 protocol reconstruction; the release has no ablation driver. Use its bowl mesh for all four materials rather than claim the Figure 4 meshes.",
    )
    prepare(args.output, config, [args.data / "mesh/bowl/bowl.obj_.msh"], args.resume)
    torch.manual_seed(0)
    torch.mps.manual_seed(0)
    v, t = read_mesh(args.data / "mesh/bowl/bowl.obj")
    fem = Elasticity(v, t, 2)
    early = SpectralLoss([2048, 1024], "transport")
    late = SpectralLoss([1024, 512, 256, 128, 64], "l1")
    rmse = SpectralLoss([1024, 512, 256, 128, 64], "rmse")
    force = torch.zeros(1, 150, device=DEVICE)
    force[:, 0] = 1
    for name, factor in (("Ceramic", 0.5), ("Iron", 2.0), ("Steel", 1.5), ("Glass", 0.7)):
        mat = MATERIALS[name]
        values, _ = fem.solve(mat[1], mat[2], mat[0], config["modes"])
        base = values.float().to(DEVICE)

        def audio(scale):
            values = base * scale
            return render(
                values.sqrt() / (2 * torch.pi),
                0.5 * (mat[3] + mat[4] * values),
                torch.ones(1, config["modes"], 1, device=DEVICE),
                force,
                config["samples"],
            )

        target, _ = audio(factor)
        for method in ("spectral", "transport", "hybrid"):
            directory = args.output / f"{name.lower()}-{method}"
            if (directory / "result.json").exists():
                continue
            model = nn.ParameterDict({"log_scale": nn.Parameter(torch.zeros((), device=DEVICE))})

            def closure(i, cache):
                signal, frequency = audio(model["log_scale"].exp())
                loss_fn = early if method == "transport" or (method == "hybrid" and i < args.early_steps) else late
                return loss_fn(signal, target, frequency), dict(scale=float(model["log_scale"].detach().exp()))

            state = train(
                directory,
                "fit",
                model,
                args.steps,
                closure,
                torch.optim.Adam(model.parameters(), lr=config["learning_rate"]),
            )
            with torch.no_grad():
                prediction, _ = audio(model["log_scale"].exp())
            save_result(
                directory,
                dict(
                    target_scale=factor,
                    estimated_scale=float(model["log_scale"].detach().exp()),
                    rmse=float(rmse(prediction, target)),
                    steps=state["step"],
                    complete_paper_reproduction=False,
                    reconstruction=config["reconstruction"],
                ),
                audio=prediction,
                target=target,
            )


if __name__ == "__main__":
    main()
