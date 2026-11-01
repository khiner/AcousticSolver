"""Compare native loss, discrete gradients, and Adam updates to captured autograd."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import reference as ref
from .native import SOURCE, SCRIPT
from .render import Stream
from .training import PARAMETERS, Trainer, flatten
from .training_reference import fresh_state, load_fixture


def error(actual, expected):
    actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
    diff = actual - expected
    return dict(
        relative_l2=float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp_min(1e-30)),
        max_absolute=float(diff.abs().max()),
    )


def compare_update(actual, ps, state, expected, before, reference_before=None):
    after = flatten(ps)
    reference_before = before if reference_before is None else reference_before
    row = dict(
        loss=actual["loss"],
        reference_loss=expected["loss"],
        loss_relative_error=abs(actual["loss"] - expected["loss"]) / max(abs(expected["loss"]), 1e-30),
        gradient=error(actual["gradient"], expected["gradient"]),
        parameters=error(after, expected["parameters_after"]),
        update=error(after - before, expected["parameters_after"] - reference_before),
        first_moment=error(state["first"], expected["first_after"]),
        second_moment=error(state["second"], expected["second_after"]),
    )
    sizes = [p.numel() for p in ps]
    row["parameter_gradients"] = {
        name: error(a, b)
        for name, a, b in zip(PARAMETERS, actual["gradient"].split(sizes), expected["gradient"].split(sizes))
    }
    row["passed"] = (
        state["step"] == expected["step"]
        and row["loss_relative_error"] < 1e-6
        and all(
            row[k]["relative_l2"] < 1e-6 for k in ("gradient", "parameters", "update", "first_moment", "second_moment")
        )
    )
    row["passed"] &= all(v["relative_l2"] < 1e-6 for v in row["parameter_gradients"].values())
    return row


def check_stream(ps, batch, expected):
    excitation = torch.cat((batch["excitation"], batch["excitation"][:, :3]), dim=1)

    def replay(initial, block):
        with Stream(ps, dict(batch, initial=initial)) as stream:
            assert stream.process(excitation[:, :0]).shape == (len(initial), 151, 0)
            return torch.cat([stream.process(x) for x in excitation.split(block, dim=1)], dim=-1)

    whole = replay(batch["initial"], excitation.shape[1])
    initial = whole[:, :, 0].clone()
    initial[:, 150] *= 1.1
    explicit = replay(initial, excitation.shape[1])
    row = dict(
        state=error(whole[:, :, : batch["excitation"].shape[1]], expected),
        block_partition_exact=torch.equal(replay(batch["initial"], 1), whole),
        explicit_psi_exact=torch.equal(explicit[:, :, 0], initial) and torch.equal(replay(initial, 17), explicit),
    )
    row["passed"] = row["state"]["relative_l2"] < 1e-6 and row["block_partition_exact"] and row["explicit_psi_exact"]
    return row


def run(args):
    torch.set_num_threads(1)
    fixture = load_fixture(args.golden)
    if tuple(fixture["parameter_order"]) != PARAMETERS or not fixture["steps"]:
        raise ValueError("Expected nonempty author updates in the supported parameter order")
    args.output.mkdir(parents=True, exist_ok=False)
    ps, state = fresh_state(fixture)
    report = dict(
        passed=False,
        backend="accelerate",
        reference_device=fixture["device"],
        golden_sha256=ref.digest(args.golden),
        reference_provenance=fixture["provenance"],
        harness={p.name: ref.digest(p) for p in SCRIPT.glob("*.py")},
        native_sources={"TrainAccelerate.cpp": ref.digest(SOURCE / "TrainAccelerate.cpp")},
        segments=len(fixture["metal_batch"]["initial"]),
        frames=fixture["metal_batch"]["target"].shape[-1],
        relative_tolerance=1e-6,
        steps=[],
    )
    trainer = Trainer()
    reference_before = flatten(ps)
    for expected in fixture["steps"]:
        before = flatten(ps)
        forward = trainer.evaluate(ps, fixture["metal_batch"], backward=False)
        streaming = check_stream(ps, fixture["metal_batch"], expected["output"])
        started = time.perf_counter()
        actual = trainer.evaluate(ps, fixture["metal_batch"])
        trainer.adam(ps, actual["gradient"], state, **fixture["adam_settings"])
        elapsed = time.perf_counter() - started
        row = compare_update(actual, ps, state, expected, before, reference_before)
        row.update(
            step=expected["step"],
            streaming=streaming,
            state=error(actual["output"], expected["output"]),
            initial_gradient=error(actual["initial_gradient"], expected["initial_gradient"]),
            loss_only_exact=forward["loss"] == actual["loss"]
            and torch.equal(forward["output"], actual["output"])
            and forward["gradient"] is None
            and forward["initial_gradient"] is None,
            total_seconds=elapsed,
        )
        row["passed"] &= (
            streaming["passed"]
            and row["loss_only_exact"]
            and all(row[k]["relative_l2"] < 1e-6 for k in ("state", "initial_gradient"))
        )
        reference_before = expected["parameters_after"]
        report["steps"].append(row)
        ref.write_json(args.output / "report.json", report)
        print(row, flush=True)
    report["passed"] = all(row["passed"] for row in report["steps"])
    ref.write_json(args.output / "report.json", report)
    np.savez(args.output / "final.npz", parameters=flatten(ps), first=state["first"], second=state["second"])
    if not report["passed"]:
        raise RuntimeError("Native training comparison failed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, help="Override the bundled CUDA fixture suite")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.golden:
        run(args)
    else:
        directory = ref.ROOT / "tests/fixtures/stable_modal_synthesis"
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = {}
        for name, record in manifest["fixtures"].items():
            path = directory / name
            if ref.digest(path) != record["sha256"]:
                raise RuntimeError(f"Fixture checksum mismatch: {name}")
            row = run(argparse.Namespace(golden=path, output=args.output / name.removesuffix(".pt.gz")))
            rows[name] = dict(
                passed=row["passed"],
                steps=len(row["steps"]),
                max_gradient_relative_error=max(s["gradient"]["relative_l2"] for s in row["steps"]),
                max_update_relative_error=max(s["update"]["relative_l2"] for s in row["steps"]),
            )
        ref.write_json(
            args.output / "report.json",
            dict(backend="accelerate", passed=all(r["passed"] for r in rows.values()), cases=rows),
        )
