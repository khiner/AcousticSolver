"""Compare distinct-recording updates to author autograd and certify exact resume."""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from torch.utils.data.distributed import DistributedSampler

from . import reference as ref
from .local_data import DEFAULT_DATA, save
from .local_train import DEFAULT_FIXTURE, Run
from .training import flatten, parameters
from .training_check import compare_update
from .training_reference import load_fixture, optimizer_result


def author_loss(model, batch, loss_func, backward=False, chunk=64):
    count, value = len(batch["y0"]), 0.0
    with torch.set_grad_enabled(backward):
        for offset in range(0, count, chunk):
            end = min(offset + chunk, count)
            part = {k: v[offset:end] for k, v in batch.items()}
            loss = loss_func({**part, **model(**part)})["loss"] * ((end - offset) / count)
            if backward:
                loss.backward()
            value += loss.item()
    return value


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def check(args):
    if args.updates < 3:
        raise ValueError("Require at least three updates to check continuation")
    args.output.mkdir(parents=True, exist_ok=False)
    run = Run(args.data, args.fixture, args.output / "continuous", args.reference)
    if args.updates <= len(run.batches("train")):
        raise ValueError("Check at least one complete epoch and one subsequent update")
    fixture = load_fixture(args.fixture)
    model = ref.learned(run.cfg)
    model.load_state_dict(fixture["model_state"])
    optimizer = instantiate(run.cfg.optimiser, params=model.parameters())
    optimizer.load_state_dict(ref.cpu(fixture["optimizer_state"]))
    collate, loss_func = instantiate(run.cfg.collate_func), instantiate(run.cfg.loss_func)
    report = dict(
        passed=False,
        contract=run.contract,
        steps=[],
        scope="Bounded updates on distinct local recordings; CPU author-autograd comparison, not trained-quality evidence",
    )
    for _ in range(args.updates):
        samplers = [
            DistributedSampler(run.datasets["train"], num_replicas=4, rank=rank, shuffle=True, seed=0)
            for rank in range(4)
        ]
        for sampler in samplers:
            sampler.set_epoch(run.epoch)
        groups = [list(group) for group in zip(*(list(s) for s in samplers))]
        assert groups == run.batches("train"), "Local batches differ from simultaneous author rank batches"
        batch = ref.double_batch(collate([run.datasets["train"][i] for i in groups[run.batch]]))
        before, reference_before = flatten(parameters(run.model)), flatten(parameters(model))
        optimizer.zero_grad(set_to_none=True)
        loss = author_loss(model, batch, loss_func, backward=True)
        gradient = flatten([p.grad for p in parameters(model)])
        optimizer.step()
        expected = dict(loss=loss, gradient=gradient, **optimizer_result(model, optimizer))
        save(args.output / f'author-step-{len(report["steps"])+1}.pt', expected)
        row, actual = run.step()
        comparison = compare_update(actual, parameters(run.model), run.adam, expected, before, reference_before)
        if "validation_loss" in row:
            values = []
            for indices in run.batches("valid"):
                valid_batch = ref.double_batch(collate([run.datasets["valid"][i] for i in indices]))
                values.append((len(indices), author_loss(model, valid_batch, loss_func)))
            valid = sum(n * v for n, v in values) / sum(n for n, _ in values)
            comparison["validation_relative_error"] = abs(row["validation_loss"] - valid) / max(abs(valid), 1e-30)
            comparison["passed"] &= comparison["validation_relative_error"] < 1e-6
        report["steps"].append(dict(update=row, comparison=comparison))
        ref.write_json(args.output / "report.json", report)
        if not comparison["passed"]:
            raise RuntimeError("Local update or validation differs from author autograd")
        print(dict(author_comparison_passed=True, step=row["optimizer_step"]), flush=True)
    split = Run(args.data, args.fixture, args.output / "resumed", args.reference)
    split.step()
    del split
    subprocess.run(
        [
            sys.executable,
            "-m",
            "script.stable_modal_synthesis.local_train",
            "--data",
            str(args.data),
            "--fixture",
            str(args.fixture),
            "--reference",
            str(args.reference),
            "--output",
            str(args.output / "resumed"),
            "--updates",
            str(args.updates - 1),
            "--resume",
            str(args.output / "resumed/last.pt"),
        ],
        check=True,
    )
    report["resume_exact"] = {
        name: exact(
            torch.load(args.output / "continuous" / name, weights_only=True),
            torch.load(args.output / "resumed" / name, weights_only=True),
        )
        for name in ("last.pt", "best.pt")
    }
    try:
        Run(
            args.data,
            args.fixture,
            args.output / "resumed",
            args.reference,
            chunk=128,
            resume=args.output / "resumed/last.pt",
        )
    except ValueError as error:
        report["changed_chunk_rejected"] = "Resume requires identical" in str(error)
    else:
        report["changed_chunk_rejected"] = False
    report["passed"] = all(report["resume_exact"].values()) and report["changed_chunk_rejected"]
    ref.write_json(args.output / "report.json", report)
    if not report["passed"]:
        raise RuntimeError("Resumed execution changed checkpoint state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--reference", type=Path, default=ref.DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    check(parser.parse_args())
