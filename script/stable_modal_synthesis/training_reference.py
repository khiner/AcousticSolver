"""Capture unchanged author-autograd updates on fixed teacher-forced windows."""

import argparse
import gzip
import time
from pathlib import Path

import torch
from hydra.utils import instantiate

from . import reference as ref
from .training import PARAMETERS, flatten, parameters, training_inputs


def load_fixture(path):
    with gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb") as f:
        return torch.load(f, weights_only=False, map_location="cpu")


def save_fixture(path, fixture):
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
            torch.save(fixture, f)


def fresh_state(fixture):
    return ([p.clone() for p in fixture["raw_parameters"]], ref.cpu(fixture["adam"]))


def adam_state(model, optimizer):
    ps = parameters(model)
    if not optimizer.state:
        count = sum(p.numel() for p in ps)
        return dict(
            step=0, first=torch.zeros(count, dtype=torch.float64), second=torch.zeros(count, dtype=torch.float64)
        )
    rows = [optimizer.state[p] for p in ps]
    steps = {int(row["step"]) for row in rows}
    if len(steps) != 1:
        raise ValueError("All parameter Adam steps must agree")
    return dict(
        step=steps.pop(),
        first=flatten([row["exp_avg"] for row in rows]),
        second=flatten([row["exp_avg_sq"] for row in rows]),
    )


def optimizer_result(model, optimizer):
    state = adam_state(model, optimizer)
    return dict(
        parameters_after=flatten(parameters(model)),
        first_after=state["first"],
        second_after=state["second"],
        step=state["step"],
    )


def make_batch(case, windows, model):
    from src.datasets import SliceCollator

    sample = torch.load(case, weights_only=False, map_location="cpu")
    batch = ref.double_batch(
        SliceCollator(0.001)(
            [dict(sample["metadata"], target_output=sample["target_output"], target_w=sample["target_w"])]
        )
    )
    count = len(batch["y0"])
    if windows is not None:
        batch = {
            k: (
                v[windows].clone()
                if isinstance(v, torch.Tensor) and v.ndim and len(v) == count
                else [v[i] for i in windows] if isinstance(v, list) and len(v) == count else v
            )
            for k, v in batch.items()
        }
    _, _, batch["phi_e"], batch["excitation"] = ref.prepared_inputs(model, batch)
    return batch


def prediction(model, batch):
    from src.ode import InitialValueProblem

    fs, frames, phi, excitation = ref.prepared_inputs(model, batch)
    return model._solver(
        InitialValueProblem(fs, frames, batch["y0"], batch["omega"], batch["sigma"], batch["mu"], phi, excitation)
    )


def capture(args):
    if args.steps < 1 or not args.windows or min(args.windows) < 0:
        raise ValueError("Require positive update count and nonnegative window indices")
    cfg = ref.load(args.reference)
    from src.losses import MSEPosVelLoss

    args.output.mkdir(parents=True, exist_ok=False)
    if args.inputs:
        fixture = load_fixture(args.inputs)
        model = ref.learned(cfg)
        model.load_state_dict(fixture["model_state"])
        batch = fixture["batch"]
    else:
        model = ref.learned(cfg, args.checkpoint)
        batch = make_batch(args.case, args.windows, model)
    model = model.to(args.device)
    optimizer = instantiate(cfg.optimiser, params=model.parameters())
    if args.inputs:
        optimizer.load_state_dict(ref.cpu(fixture["optimizer_state"]))
    elif args.restore_optimizer:
        if args.checkpoint is None:
            raise ValueError("--restore-optimizer requires --checkpoint")
        optimizer.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)["optimiser_state_dict"]
        )
    if not args.inputs:
        fixture = dict(
            model_state=ref.cpu(model.state_dict()),
            optimizer_state=ref.cpu(optimizer.state_dict()),
            batch=ref.cpu(batch),
            windows=args.windows,
            source_case=args.case.name,
            source_case_sha256=ref.digest(args.case),
            checkpoint=args.checkpoint.name if args.checkpoint else None,
            checkpoint_sha256=ref.digest(args.checkpoint) if args.checkpoint else None,
            restored_optimizer=args.restore_optimizer,
        )
    (group,) = optimizer.param_groups
    if group["weight_decay"] or group["amsgrad"] or group.get("maximize", False):
        raise ValueError("Only the released unregularized Adam settings are supported")
    fixture["adam_settings"] = dict(lr=group["lr"], betas=group["betas"], epsilon=group["eps"])
    fixture["raw_parameters"] = ref.cpu(parameters(model))
    fixture["metal_batch"] = ref.cpu(training_inputs(model, batch))
    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch["y0"] = batch["y0"].detach().requires_grad_(True)
    fixture["adam"] = adam_state(model, optimizer)
    fixture["steps"] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        batch["y0"].grad = None
        started = time.perf_counter()
        output = prediction(model, batch)
        loss = MSEPosVelLoss()({**batch, "output": output})["loss"]
        loss.backward()
        gradient = flatten([p.grad for p in parameters(model)])
        row = dict(
            loss=loss.item(), output=ref.cpu(output), gradient=gradient, initial_gradient=ref.cpu(batch["y0"].grad)
        )
        optimizer.step()
        row.update(optimizer_result(model, optimizer), seconds=time.perf_counter() - started)
        fixture["steps"].append(row)
        print(dict(update=step + 1, optimizer_step=row["step"], loss=row["loss"], seconds=row["seconds"]), flush=True)
    fixture["parameter_order"] = PARAMETERS
    fixture["device"] = args.device
    provenance = ref.provenance(args.reference)
    provenance.update(device=args.device, device_name=torch.cuda.get_device_name() if args.device == "cuda" else "CPU")
    fixture["provenance"] = provenance
    save_fixture(args.output / "reference.pt.gz", fixture)
    ref.write_json(
        args.output / "report.json",
        dict(
            provenance=provenance,
            fixture_sha256=ref.digest(args.output / "reference.pt.gz"),
            windows=fixture["windows"],
            segments=len(batch["y0"]),
            samples=batch["target_output"].shape[-1],
            steps=args.steps,
            scope="Fixed-window derivative and optimizer comparison, not a full training batch or paper-quality reproduction",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ref.DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--case", type=Path, default=ref.ROOT / "build/stable-modal-cuda/results/author-checkpoint/train-case-0.pt"
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[0, 1, 20, 1000])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--restore-optimizer", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--inputs", type=Path, help="Reuse an existing reference.pt input and optimizer fixture on another device"
    )
    capture(parser.parse_args())
