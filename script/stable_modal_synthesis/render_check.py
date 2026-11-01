"""Compare native forward synthesis against the pinned float64 solver on full trajectories."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import default_collate

from . import reference as ref
from .metal import Renderer, SHADER_SOURCES
from .native import SOURCE
from .render import Stream, render
from .training import parameters, synthesis_inputs


def relative(actual, expected):
    return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))


def run(args):
    cfg = ref.load(args.reference, args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        passed=False,
        provenance=ref.provenance(args.reference),
        cases=[],
        native_sources={p.name: ref.digest(p) for p in (*SOURCE.glob("*.cpp"), *SHADER_SOURCES)},
        waveform_relative_tolerance=1e-6,
        state_relative_tolerance=1e-6,
        scope="Complete native forward trajectories; gradients are validated separately by training_check",
    )
    with Renderer() as renderer:
        golden_report = json.loads((args.golden / "report.json").read_text()) if args.golden else None
        if args.golden:
            report["golden"] = dict(
                report=golden_report, files={p.name: ref.digest(p) for p in args.golden.glob("*.pt")}
            )
        for split in (golden_report["split"],) if args.golden else ("train", "test"):
            config = cfg.datasets[split]
            batch = (
                torch.load(args.golden / "batch.pt", weights_only=True)
                if args.golden
                else default_collate(ref.metadata(config)[:1])
            )
            duration = golden_report["frames"] / config.fs if args.golden else args.seconds or config.dur
            for kind in ("spectral", "learned"):
                batch["dur"] = torch.tensor([duration], dtype=torch.float64)
                if kind == "learned":
                    model = ref.learned(cfg, args.reference / ref.PROTOCOL["checkpoint"])
                else:
                    model = ref.generator(config, duration=duration)
                    batch["dur"] = model._dur.clone()
                with torch.no_grad():
                    start = time.perf_counter()
                    if args.golden:
                        expected = torch.load(
                            args.golden / ("prediction.pt" if kind == "learned" else "target.pt"), weights_only=True
                        )
                    else:
                        expected = model(**batch) if kind == "learned" else ref.generate(model, batch)
                    reference_seconds = time.perf_counter() - start
                    actual = render(model, batch) if kind == "learned" else renderer.render(model, batch)
                metrics = dict(
                    split=split,
                    kind=kind,
                    duration=duration,
                    sample_rate=config.fs,
                    waveform_relative_error=relative(actual["w"], expected["w"]),
                    q_relative_error=relative(actual["output"][:, :75], expected["output"][:, :75]),
                    p_relative_error=relative(actual["output"][:, 75:150], expected["output"][:, 75:150]),
                    psi_relative_error=relative(actual["output"][:, 150:], expected["output"][:, 150:]),
                    **({} if args.golden else dict(reference_seconds=reference_seconds)),
                    **actual["timing"]
                )
                metrics["passed"] = all(
                    metrics[k] < 1e-6
                    for k in ("waveform_relative_error", "q_relative_error", "p_relative_error", "psi_relative_error")
                )
                prefix = args.output / (split + "-" + kind)
                sf.write(str(prefix) + "-reference.wav", expected["w"][0].numpy(), config.fs, subtype="DOUBLE")
                sf.write(str(prefix) + "-native.wav", actual["w"][0].numpy(), config.fs, subtype="DOUBLE")
                np.savez(
                    str(prefix) + "-final.npz",
                    reference=expected["output"][0, :, -1].numpy(),
                    native=actual["output"][0, :, -1].numpy(),
                )
                report["cases"].append(metrics)
                ref.write_json(args.output / "report.json", report)
                print(metrics, flush=True)
        prepared = synthesis_inputs(model, batch)
        excitation = prepared["excitation"]
        partition_exact = {}
        for block_size in (127, 1024):
            with Stream(parameters(model), prepared) as stream:
                exact = True
                for first in range(0, excitation.shape[1], block_size):
                    part = stream.process(excitation[:, first : first + block_size])
                    exact &= torch.equal(part, actual["output"][:, :, first : first + part.shape[-1]])
            partition_exact[str(block_size)] = exact
        report["block_partition_exact"] = partition_exact
        report["passed"] = all(c["passed"] for c in report["cases"]) and all(partition_exact.values())
        ref.write_json(args.output / "report.json", report)
        if not report["passed"]:
            raise RuntimeError("Native forward reference comparison failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ref.DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, help="Explicit shortened diagnostic instead of paper durations")
    parser.add_argument("--golden", type=Path, help="Compare exact saved CUDA inputs and outputs from golden.py")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.golden and args.seconds:
        parser.error("--golden and --seconds cannot be combined")
    run(args)
