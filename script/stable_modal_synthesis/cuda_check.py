"""Measure one full training batch and verify CUDA Graph inference equivalence."""

import argparse
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from torch.utils.data import default_collate

from . import reference as ref


def run(args):
    cfg = ref.load(args.reference)
    from .cuda_graph import install
    from src.datasets import SliceCollator
    from src.losses import MSEPosVelLoss

    metadata = ref.metadata(cfg.datasets.train, "cuda")[:4]
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in default_collate(metadata).items()}
    short = ref.generator(cfg.datasets.train, duration=0.02).cuda()
    learned = ref.learned(cfg, args.reference / ref.PROTOCOL["checkpoint"]).cuda()
    batch["dur"] = short._dur
    with torch.inference_mode():
        original_target = ref.generate(short, batch)
        original_prediction = learned(**batch)
        install()
        captured_target = ref.generate(short, batch)
        captured_prediction = learned(**batch)
    exact = {
        name: {key: torch.equal(original[key], captured[key]) for key in ("output", "w")}
        for name, original, captured in (
            ("target", original_target, captured_target),
            ("prediction", original_prediction, captured_prediction),
        )
    }
    if not all(all(row.values()) for row in exact.values()):
        raise RuntimeError(f"CUDA Graph changed output: {exact}")
    print(dict(cuda_graph_exact=exact), flush=True)
    del short, learned, original_target, original_prediction, captured_target, captured_prediction
    generator = ref.generator(cfg.datasets.train).cuda()
    batch["dur"] = generator._dur
    with torch.inference_mode():
        target = ref.generate(generator, batch)
    instances = [
        dict(row, target_output=target["output"][i].cpu(), target_w=target["w"][i].cpu())
        for i, row in enumerate(metadata)
    ]
    del target, generator, batch
    sliced = ref.double_batch(SliceCollator(0.001)(instances))
    del instances
    sliced = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in sliced.items()}
    model = ref.learned(cfg).cuda()
    optimiser = instantiate(cfg.optimiser, params=model.parameters())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for step in range(3):
        optimiser.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = model(**sliced)
        loss = MSEPosVelLoss()({**sliced, **prediction})["loss"]
        loss.backward()
        finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters())
        if not finite:
            raise RuntimeError("Nonfinite training gradient")
        optimiser.step()
        torch.cuda.synchronize()
        row = dict(
            step=step,
            seconds=time.perf_counter() - started,
            loss=loss.item(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        rows.append(row)
        print(row, flush=True)
        del prediction, loss
    provenance = ref.provenance(args.reference)
    provenance["device"] = torch.cuda.get_device_name()
    ref.write_json(
        args.output,
        dict(
            provenance=provenance,
            cuda_graph_exact=exact,
            steps=rows,
            recordings=4,
            segments=len(sliced["y0"]),
            samples=sliced["target_output"].shape[-1],
            scope="Three updates on four training trajectories; execution and memory diagnostic only",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ref.DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
