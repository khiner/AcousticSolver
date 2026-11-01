"""Run DiffSound's released physical inference tasks locally."""

import argparse
import json
from pathlib import Path

import torch

from . import material, shape
from .runtime import prepare

DEFAULTS = {
    "material-synthetic": dict(
        rate=32000,
        samples=8000,
        force_samples=150,
        modes=16,
        steps=3000,
        early_steps=1000,
        exp_mode=3,
        material_init_steps=5000,
        cases=16,
    ),
    "material-real": dict(
        rate=32000,
        samples=8000,
        force_samples=150,
        modes=16,
        steps=3000,
        early_steps=1000,
        exp_mode=3,
        material_init_steps=5000,
        damping_steps=2001,
        recordings=8,
        material="Ceramic",
    ),
    "thickness": dict(
        grid=64,
        generation_grid=128,
        scale=1.5,
        first="frog",
        targets=[0.3, 0.4, 0.5, 0.6, 0.7],
        modes=32,
        steps=500,
        learning_rate=0.02,
        material="Steel",
    ),
    "morphing": dict(
        grid=64,
        generation_grid=128,
        scale=1.5,
        first="frog",
        second="turtle",
        targets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        modes=16,
        steps=10,
        learning_rate=0.02,
        material="Steel",
    ),
    "geometry": dict(
        grid=32,
        voxels=16,
        frequencies=3,
        shapes=["bob", "oloid", "spot"],
        modes=64,
        steps=1000,
        pretrain_steps=2000,
        learning_rate=1e-5,
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=DEFAULTS)
    parser.add_argument("--data", type=Path, default=Path("data/diffsound/source/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON overrides; every effective setting is saved")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = dict(DEFAULTS[args.task])
    if args.config:
        updates = json.loads(args.config.read_text())
        if set(updates) - set(config):
            raise ValueError(f"Unknown configuration keys: {set(updates)-set(config)}")
        config.update(updates)
    config["seed"] = args.seed
    config["task"] = args.task
    if config["steps"] < 1:
        raise ValueError("steps must be positive")
    sources = sorted(p for p in args.data.rglob("*") if p.is_file())
    if not sources:
        raise FileNotFoundError(args.data)
    prepare(args.output, config, sources, args.resume)
    torch.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)
    if args.task.startswith("material"):
        material.run(args.data, args.output, config, args.task == "material-real")
    elif args.task == "geometry":
        shape.reconstruct(args.data, args.output, config)
    else:
        shape.scalar(args.data, args.output, config, args.task)


if __name__ == "__main__":
    main()
