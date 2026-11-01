"""Capture one CUDA float64 trajectory and its exact prepared inputs for Metal."""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import default_collate

from . import reference as ref


def run(args):
    cfg = ref.load(args.reference)
    if args.cuda_graph:
        from .cuda_graph import install

        install()
    args.output.mkdir(parents=True, exist_ok=False)
    config = cfg.datasets[args.split]
    batch = default_collate([ref.metadata(config, args.device)[args.index]])
    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model = ref.learned(cfg, args.reference / ref.PROTOCOL["checkpoint"]).to(args.device)
    gen = ref.generator(config, duration=args.seconds).to(args.device)
    batch["dur"] = gen._dur
    started = time.perf_counter()
    with torch.inference_mode():
        target = ref.generate(gen, batch)
        print("Target generated", flush=True)
        prediction = model(**batch)
        print("Checkpoint rendered", flush=True)
        _, _, batch["phi_e"], batch["excitation"] = ref.prepared_inputs(model, batch)
        batch["phi_o"] = model._Phi(batch["xo"], 75)
        for name, values in (("batch", batch), ("target", target), ("prediction", prediction)):
            torch.save(ref.cpu(values), args.output / (name + ".pt"))
    errors = {}
    for name, result, suffix in (("audio", target, ""), ("prediction", prediction, "_pred")):
        audio = result["w"][0].cpu().numpy()
        published, _ = sf.read(
            args.reference / "audio" / ref.PROTOCOL["audio_directories"][args.split] / f"{args.index}{suffix}.wav"
        )
        published = published[: len(audio)]
        errors[f"published_{name}_relative_error"] = float(
            np.linalg.norm(audio / np.max(np.abs(audio)) - published) / np.linalg.norm(published)
        )
    provenance = ref.provenance(args.reference)
    provenance["device"] = args.device
    report = dict(
        device=args.device,
        cuda_graph=args.cuda_graph,
        split=args.split,
        index=args.index,
        frames=len(audio),
        seconds=time.perf_counter() - started,
        **errors,
        published_audio_normalization="full recording peak; truncated captures may have a different peak",
        provenance=provenance,
    )
    ref.write_json(args.output / "report.json", report)
    print({k: v for k, v in report.items() if k != "provenance"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ref.DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--cuda-graph", action="store_true")
    run(parser.parse_args())
