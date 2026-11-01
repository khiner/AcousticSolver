"""Fit DiffImpact or train its audio encoder using one MLX/Metal implementation."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import signal
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from scipy.io import wavfile

from . import data, dsp, models


class Adam:
    """Legacy Keras Adam, including per-variable clipping and epsilon placement."""

    def __init__(self):
        self.step = 0
        self.m, self.v = {}, {}

    def update(self, model, gradients):
        iteration = self.step + 1
        rate = 0.001 * math.sqrt(1 - 0.999**iteration) / (1 - 0.9**iteration)
        parameters = dict(tree_flatten(model.trainable_parameters()))
        result = []
        for name, g in tree_flatten(gradients):
            g = g / mx.maximum(1, mx.sqrt(mx.sum(g * g)))
            m, v = self.m.get(name, mx.zeros_like(g)), self.v.get(name, mx.zeros_like(g))
            self.m[name] = m + (g - m) * 0.1
            self.v[name] = v + (g * g - v) * 0.001
            value = parameters[name] - rate * self.m[name] / (mx.sqrt(self.v[name]) + 1e-7)
            if name in ("acceleration_scale", "noise_scale", "mag", "tau", "peak_time"):
                value = mx.maximum(value, 0)
            result.append((name, value))
        model.update(tree_unflatten(result))
        self.step = iteration


def save(path, model, optimizer, metadata):
    arrays = {"parameter." + k: np.asarray(v) for k, v in tree_flatten(model.parameters())}
    arrays.update({"adam_m." + k: np.asarray(v) for k, v in optimizer.m.items()})
    arrays.update({"adam_v." + k: np.asarray(v) for k, v in optimizer.v.items()})
    arrays["metadata"] = np.array(json.dumps(dict(metadata, step=optimizer.step)))
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez(file, **arrays)
    temporary.replace(path)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["experiment"] == "properties":
            from .properties import PropertyModel

            model = PropertyModel(metadata["config"])
        else:
            model = models.MODELS[metadata["experiment"]](metadata["config"])
        model.freeze(keys=metadata.get("frozen_parameters", []), strict=True)
        parameters = {k[10:]: mx.array(archive[k]) for k in archive.files if k.startswith("parameter.")}
        expected = dict(tree_flatten(model.parameters()))
        if parameters.keys() != expected.keys() or any(parameters[k].shape != expected[k].shape for k in expected):
            raise ValueError("Checkpoint architecture does not match its configuration")
        model.update(tree_unflatten(list(parameters.items())))
        optimizer = Adam()
        optimizer.step = metadata["step"]
        optimizer.m = {k[7:]: mx.array(archive[k]) for k in archive.files if k.startswith("adam_m.")}
        optimizer.v = {k[7:]: mx.array(archive[k]) for k in archive.files if k.startswith("adam_v.")}
        trainable = dict(tree_flatten(model.trainable_parameters()))
        if optimizer.step and any(
            slots.keys() != trainable.keys() or any(slots[k].shape != trainable[k].shape for k in trainable)
            for slots in (optimizer.m, optimizer.v)
        ):
            raise ValueError("Checkpoint optimizer state is incomplete")
    return model, optimizer, metadata


def objective(model, batch, draws):
    c = model._config
    return dsp.spectral_loss(batch["audio"], model(batch, draws)["audio"], c["fft_sizes"], c["log_weight"])


def random_key(seed, index):
    return mx.array(np.random.SeedSequence([seed, index]).generate_state(2))


def step(model, optimizer, batch, seed):
    key = random_key(seed, optimizer.step)
    draws = model.noise(len(batch["audio"]), key)
    loss, gradients = nn.value_and_grad(model, objective)(model, batch, draws)
    finite = mx.all(mx.stack([mx.all(mx.isfinite(g)) for _, g in tree_flatten(gradients)]))
    mx.eval(loss, gradients, finite)
    if not bool(finite) or not math.isfinite(float(loss)):
        raise FloatingPointError("Non-finite training loss/gradient; checkpoint was not advanced")
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.m, optimizer.v)
    return float(loss)


def configuration(experiment, groups, channels):
    audio = groups["train"]["audio"]
    separation = experiment == "separation"
    size = audio.shape[1]
    config = dict(
        rate=44100,
        samples=size,
        frames=round(size / 441),
        bodies=1 if experiment == "autoencoder" else 4,
        modes=128 if separation else 256,
        ir_samples=132300 if separation else 88200,
        hz_max=18000 if separation else 20000,
        reverb_samples=96000,
        fft_sizes=[2048, 1024, 512, 256] + ([] if experiment == "hammer" else [128]),
        log_weight=0.1 if separation else 1.0,
    )
    if separation:
        config.update(tools=4, recordings=len(audio), peak_times=(np.argmax(abs(audio), axis=1) / 441).tolist())
    if experiment == "autoencoder":
        config["channels"] = channels
    return config


def prepare(args):
    if args.experiment == "autoencoder":
        if args.manifest is None or args.steps is None:
            raise ValueError("Autoencoder requires an explicit manifest and step budget; the release provides neither")
        groups, sources, spec = data.asmr(args.manifest)
        for batch in groups.values():
            batch["mfcc"] = np.asarray(models.mfcc(mx.array(batch["audio"]), 44100))
    else:
        data.verify_reference(args.reference)
        loader = data.hammer if args.experiment == "hammer" else data.separation
        batch, sources = loader(args.reference)
        groups, spec = {"train": batch}, None
    config = configuration(args.experiment, groups, args.channels)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "dataset.npz", **{f"{group}.{k}": v for group, batch in groups.items() for k, v in batch.items()}
    )
    protocol_path = Path(__file__).with_name("protocol.json")
    (args.output / "protocol.json").write_bytes(protocol_path.read_bytes())
    metadata = dict(
        schema=1,
        experiment=args.experiment,
        config=config,
        seed=args.seed,
        sources=sources,
        dataset_sha256=data.digest(args.output / "dataset.npz"),
        protocol_sha256=data.digest(protocol_path),
        implementation={
            name: data.digest(Path(__file__).with_name(name)) for name in ("dsp.py", "models.py", "data.py", "train.py")
        },
        packages={name: importlib.metadata.version(name) for name in ("mlx", "numpy", "scipy")},
        asmr_manifest=spec,
        batch_size=args.batch_size,
        device=mx.device_info()["device_name"],
        validation_interval=args.checkpoint_every,
        adaptations=[
            "batch-independent contact indexing",
            "constant-frequency phase without float32 cumsum drift",
            "MLX initialization and random streams",
        ]
        + (
            [
                "released DDSP encoder/decoder with explicit ASMR wiring and default widths",
                "explicit user-supplied ASMR splits and schedule; per-clip normalization",
            ]
            + spec.get("adaptations", [])
            if spec
            else []
        ),
    )
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    mx.random.seed(args.seed)
    return models.MODELS[args.experiment](config), Adam(), metadata


def batches(dataset, count):
    for start in range(0, len(dataset["audio"]), count):
        yield {k: mx.array(v[start : start + count]) for k, v in dataset.items() if k != "measured_force"}


def predictions(model, dataset, count):
    for i, batch in enumerate(batches(dataset, count)):
        rendered = model(batch, model.noise(len(batch["audio"]), random_key(714, i)))
        value = dsp.spectral_loss(
            batch["audio"], rendered["audio"], model._config["fft_sizes"], model._config["log_weight"]
        )
        yield rendered, float(value), len(batch["audio"])


def validate(output, model, optimizer, metadata, dataset):
    history = metadata.setdefault("validation_history", [])
    if history and history[-1]["step"] == optimizer.step:
        return
    value = sum(value * count for _, value, count in predictions(model, dataset, metadata["batch_size"])) / len(
        dataset["audio"]
    )
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite validation loss; best checkpoint was not changed")
    history.append(dict(step=optimizer.step, loss=value))
    if "best_validation" not in metadata or value < metadata["best_validation"]:
        metadata.update(best_validation=value, best_step=optimizer.step)
        save(output / "best.npz", model, optimizer, metadata)
    print(json.dumps(dict(step=optimizer.step, validation=value, best_step=metadata["best_step"])), flush=True)


def export(output, model, optimizer, metadata, groups, checkpoint="checkpoint.npz"):
    results, metrics = {}, {}
    for group, dataset in groups.items():
        count = metadata["batch_size"] if metadata["experiment"] == "autoencoder" else len(dataset["audio"])
        collected, loss = {}, 0.0
        for rendered, value, batch_count in predictions(model, dataset, count):
            loss += value * batch_count
            for name, audio in rendered.items():
                collected.setdefault(name, []).append(np.asarray(audio))
        metrics[group] = loss / len(dataset["audio"])
        for name, chunks in collected.items():
            results[f"{group}.{name}"] = np.concatenate(chunks)
        for i, audio in enumerate(results[f"{group}.audio"]):
            wavfile.write(output / f"{group}-{i:03d}.wav", model._config["rate"], audio)
    np.savez_compressed(output / "rendered.npz", **results)
    physical = {}
    for prefix in ("", "tool_") if metadata["experiment"] == "separation" else ("",):
        gains, frequencies, dampings = dsp.modal_controls(
            model[prefix + "gains"], model[prefix + "frequencies"], model[prefix + "dampings"], model._config["hz_max"]
        )
        physical[prefix + "gains"] = np.asarray(gains).tolist()
        physical[prefix + "frequencies"] = np.asarray(frequencies).tolist()
        physical[prefix + "dampings"] = np.asarray(dampings).tolist()
    report = dict(
        step=optimizer.step,
        spectral_loss=metrics,
        physical=physical,
        contact_rate=model._rate,
        audio_rate=model._config["rate"],
        adaptations=metadata["adaptations"],
        complete_paper_reproduction=False,
    )
    report["checkpoint"] = checkpoint
    report["checkpoint_sha256"] = data.digest(output / checkpoint)
    if metadata["experiment"] == "autoencoder":
        with np.load(output / "checkpoint.npz", allow_pickle=False) as archive:
            latest = json.loads(str(archive["metadata"]))
        report.update(
            training_step=latest["step"],
            validation_history=latest["validation_history"],
            selection="Minimum validation spectral loss with fixed evaluation noise; test excluded from selection",
        )
    report["rendered_sha256"] = data.digest(output / "rendered.npz")
    if metadata["experiment"] == "hammer":
        measured = groups["train"]["measured_force"].astype(np.float64)
        predicted = results["train.force"][:, ::2].astype(np.float64)
        scale = np.maximum(0, np.sum(measured * predicted, axis=1) / np.maximum(1e-30, np.sum(predicted**2, axis=1)))
        report["force_relative_error_after_scale_fit"] = (
            np.linalg.norm(measured - predicted * scale[:, None], axis=1) / np.linalg.norm(measured, axis=1)
        ).tolist()
        report["force_units"] = "Uncalibrated; measured force is evaluation-only"
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=models.MODELS)
    parser.add_argument("--reference", type=Path, default=Path("external/diffimpact"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.checkpoint_every <= 0 or (args.steps is not None and args.steps < 0):
        parser.error("Invalid batch size, checkpoint interval or step count")
    if args.resume or args.evaluate:
        checkpoint = "best.npz" if args.evaluate and args.experiment == "autoencoder" else "checkpoint.npz"
        model, optimizer, metadata = load(args.output / checkpoint)
        if metadata["experiment"] != args.experiment:
            parser.error("Experiment does not match checkpoint")
    else:
        model, optimizer, metadata = prepare(args)
        save(args.output / "checkpoint.npz", model, optimizer, metadata)
        if args.experiment == "autoencoder":
            save(args.output / "initial.npz", model, optimizer, metadata)
    if data.digest(args.output / "dataset.npz") != metadata["dataset_sha256"]:
        raise ValueError("Checkpoint dataset hash mismatch")
    groups = {}
    with np.load(args.output / "dataset.npz", allow_pickle=False) as archive:
        for key in archive.files:
            group, name = key.split(".", 1)
            groups.setdefault(group, {})[name] = archive[key]
    steps = args.steps if args.steps is not None else {"hammer": 10000, "separation": 6000}.get(args.experiment)
    if not args.evaluate and steps is None:
        parser.error("Specify the autoencoder training step budget")
    if args.evaluate:
        print(
            json.dumps(
                dict(
                    step=optimizer.step,
                    spectral_loss=export(args.output, model, optimizer, metadata, groups, checkpoint),
                )
            ),
            flush=True,
        )
        return
    autoencoder = args.experiment == "autoencoder"
    interval = metadata["validation_interval"] if autoencoder else args.checkpoint_every
    if autoencoder:
        validate(args.output, model, optimizer, metadata, groups["validation"])
        save(args.output / "checkpoint.npz", model, optimizer, metadata)
    started = time.monotonic()
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        print("Stopping after the current update; checkpoint and artifacts will be saved", flush=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not stopping and optimizer.step < steps:
        dataset = groups["train"]
        if args.experiment == "autoencoder":
            rng = np.random.default_rng(np.random.SeedSequence([metadata["seed"], optimizer.step]))
            indices = rng.choice(
                len(dataset["audio"]), min(metadata["batch_size"], len(dataset["audio"])), replace=False
            )
            dataset = {k: v[indices] for k, v in dataset.items()}
        batch = next(batches(dataset, len(dataset["audio"])))
        loss = step(model, optimizer, batch, metadata["seed"])
        if optimizer.step % interval == 0 or optimizer.step == steps:
            if autoencoder:
                validate(args.output, model, optimizer, metadata, groups["validation"])
            save(args.output / "checkpoint.npz", model, optimizer, metadata)
            print(
                json.dumps(dict(step=optimizer.step, loss=loss, elapsed_seconds=time.monotonic() - started)), flush=True
            )
    save(args.output / "checkpoint.npz", model, optimizer, metadata)
    if autoencoder:
        print(
            json.dumps(
                dict(step=optimizer.step, best_step=metadata["best_step"], best_validation=metadata["best_validation"])
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                dict(step=optimizer.step, spectral_loss=export(args.output, model, optimizer, metadata, groups))
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
