"""Retain an exact prefix of a captured update sequence as a repository fixture."""

import argparse
from pathlib import Path

from . import reference as ref
from .training_reference import load_fixture, save_fixture


def pack(source, destination, steps):
    fixture = load_fixture(source)
    if fixture["device"] != "cuda" or not 1 <= steps <= len(fixture["steps"]):
        raise ValueError("Require a nonempty prefix of a CUDA capture")
    fixture["full_capture"] = dict(sha256=ref.digest(source), steps=len(fixture["steps"]))
    fixture["steps"] = fixture["steps"][:steps]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    save_fixture(destination, fixture)
    return dict(
        sha256=ref.digest(destination),
        bytes=destination.stat().st_size,
        steps=steps,
        first_optimizer_step=fixture["steps"][0]["step"],
        last_optimizer_step=fixture["steps"][-1]["step"],
        frames=fixture["metal_batch"]["target"].shape[-1],
        windows=fixture["windows"],
        full_capture=fixture["full_capture"],
        device=fixture["provenance"]["device_name"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    ref.write_json(args.destination.with_suffix(".json"), pack(args.source, args.destination, args.steps))
