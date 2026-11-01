"""Appendix D property networks, reconstructed where the release omits settings."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import signal

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
from scipy.io import wavfile

from . import data, dsp
from .train import Adam, load, random_key, save

MATERIALS = ("ceramic", "polycarb", "steel", "wood")
CONFIG = dict(channels=128, hidden=64, dropout=0.2, rate=44100, window=512, hop=128)


class PropertyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = dict(config)
        c = config["channels"]
        inputs = config["window"] // 2 + 1
        if config["task"] == "material":
            self.gru = nn.GRU(inputs, c)
        else:
            self.convs = [nn.Conv1d(inputs if i == 0 else c, c, 3, padding=2**i, dilation=2**i) for i in range(3)]
        self.dense = [nn.Linear(c, c), nn.Linear(c, config["hidden"])]
        self.head = nn.Linear(config["hidden"], 4 if config["task"] == "material" else 1)

    def __call__(self, features, lengths, key=None):
        keys = mx.random.split(key, 3) if key is not None else [None] * 3

        def dropout(x, index):
            p = self._config["dropout"]
            return x if key is None else x * mx.random.bernoulli(1 - p, x.shape, key=keys[index]) / (1 - p)

        mask = mx.arange(features.shape[1])[None, :, None] < lengths[:, None, None]
        z = mx.where(mask, features, 0)
        if self._config["task"] == "material":
            z = self.gru(z)[mx.arange(len(z)), lengths - 1]
        else:
            for conv in self.convs:
                z = mx.where(mask, nn.relu(conv(z)), 0)
            z = mx.max(mx.where(mask, z, -mx.inf), axis=1)
        z = dropout(z, 0)
        for i, layer in enumerate(self.dense):
            z = dropout(nn.relu(layer(z)), i + 1)
        z = self.head(z)
        return z if self._config["task"] == "material" else mx.exp(z[:, 0]) * 1e10


def loss(model, features, lengths, labels, key=None):
    predictions = model(features, lengths, key)
    if model._config["task"] == "material":
        return nn.losses.cross_entropy(predictions, labels, reduction="mean")
    return mx.mean((mx.log1p(predictions) - mx.log1p(labels)) ** 2)


def split(records, task, seed):
    """Stratify by material, keep each simulation together, and balance the classifier."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, item in enumerate(records):
        groups.setdefault(item["group"], []).append(i)
    materials = sorted({r["material"] for r in records})
    if task == "material" and not set(MATERIALS) <= set(materials):
        raise ValueError("Material training requires all four paper materials")
    result = {"train": [], "validation": []}
    buckets = {k: [] for k in result}
    # A mixed-material simulation must not be assigned separately for different labels.
    by_material = {}
    for name, ids in groups.items():
        labels = {records[i]["material"] for i in ids}
        if len(labels) != 1:
            raise ValueError("A group must identify one simulated object/material recording session")
        by_material.setdefault(next(iter(labels)), []).append(name)
    for material in materials:
        if task == "material" and material not in MATERIALS:
            continue
        names = rng.permutation(by_material[material])
        if len(names) < 2:
            raise ValueError(f"Need at least two independent groups for {material}")
        count = max(1, round(0.1 * len(names)))
        for partition, chosen in (("validation", names[:count]), ("train", names[count:])):
            buckets[partition].append(rng.permutation([i for name in chosen for i in groups[name]]).tolist())
    for partition, lists in buckets.items():
        count = min(map(len, lists)) if task == "material" else None
        result[partition] = [i for ids in lists for i in ids[:count]]
    return result


def features(audio, config):
    return np.asarray(
        dsp.magnitude(mx.array(data.normalize(audio)[None].astype(np.float32)), config["window"], config["hop"])
    )[0]


def batch(root, records, indices, task):
    values = [np.load(root / "features" / f"{i}.npy", allow_pickle=False) for i in indices]
    lengths = np.array([len(v) for v in values], np.int32)
    padded = np.zeros((len(values), max(lengths), values[0].shape[1]), np.float32)
    for i, value in enumerate(values):
        padded[i, : len(value)] = value
    labels = [
        MATERIALS.index(records[i]["material"]) if task == "material" else records[i]["youngs_pa"] for i in indices
    ]
    return mx.array(padded), mx.array(lengths), mx.array(labels, mx.int32 if task == "material" else mx.float32)


def prepare(args):
    spec = json.loads(args.manifest.read_text())
    records = spec["recordings"]
    if spec["dataset"] != "Sound-20K" or not records:
        raise ValueError("Appendix D training requires a nonempty Sound-20K manifest")
    if len({r["sha256"] for r in records}) != len(records):
        raise ValueError("Duplicate audio would leak between dataset splits")
    if any(not np.isfinite(r["youngs_pa"]) or r["youngs_pa"] <= 0 for r in records):
        raise ValueError("Young modulus labels must be positive finite Pascals")
    partitions = split(records, args.task, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "features").mkdir()
    config = dict(CONFIG, task=args.task)
    cached = []
    for i, item in enumerate(records):
        path = (args.manifest.parent / item["path"]).resolve()
        if data.digest(path) != item["sha256"]:
            raise ValueError(f"Sound-20K hash mismatch: {path}")
        rate, audio = wavfile.read(path)
        if rate != config["rate"] or audio.ndim != 1:
            raise ValueError(f"Expected mono 44100 Hz isolated-object track: {path}")
        feature_path = args.output / "features" / f"{i}.npy"
        np.save(feature_path, features(audio, config))
        cached.append(dict(item, path=str(path), features_sha256=data.digest(feature_path)))
    metadata = dict(
        experiment="properties",
        config=config,
        seed=args.seed,
        batch_size=args.batch_size,
        patience=args.patience,
        records=cached,
        partitions=partitions,
        manifest_sha256=data.digest(args.manifest),
        epoch=0,
        cursor=0,
        best_validation=None,
        stale=0,
        complete_paper_reproduction=False,
        provenance=spec.get("provenance"),
        packages={name: importlib.metadata.version(name) for name in ("mlx", "numpy", "scipy")},
        protocol_sha256=data.digest(Path(__file__).with_name("protocol.json")),
        implementation={
            name: data.digest(Path(__file__).with_name(name)) for name in ("properties.py", "train.py", "dsp.py")
        },
        adaptations=[
            "Appendix D architecture with unreleased widths and dropout reconstructed",
            "periodic Hann, hop 128, mean centering and peak normalization",
            "seeded grouped stratified 90/10 split; balance each classifier split",
            "legacy clipped Adam from released fitting code",
            "log-Pascal exponential regressor head; MSLE in Pascals",
        ],
    )
    mx.random.seed(args.seed)
    return PropertyModel(config), Adam(), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("material", "youngs"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) <= 0 or (not args.resume and args.manifest is None):
        parser.error("Positive budgets and a manifest for a new run are required")
    model, optimizer, meta = load(args.output / "checkpoint.npz") if args.resume else prepare(args)
    if meta["experiment"] != "properties" or meta["config"]["task"] != args.task:
        parser.error("Checkpoint task mismatch")
    for i, item in enumerate(meta["records"]):
        if data.digest(args.output / "features" / f"{i}.npy") != item["features_sha256"]:
            raise ValueError("Cached dataset hash mismatch")
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    save(args.output / "checkpoint.npz", model, optimizer, meta)
    value_and_grad = nn.value_and_grad(model, loss)
    while meta["epoch"] < args.epochs and meta["stale"] < meta["patience"] and not stopping:
        rng = np.random.default_rng(np.random.SeedSequence([meta["seed"], meta["epoch"]]))
        indices = rng.permutation(meta["partitions"]["train"])
        while meta["cursor"] < len(indices) and not stopping:
            chosen = indices[meta["cursor"] : meta["cursor"] + meta["batch_size"]]
            inputs = batch(args.output, meta["records"], chosen, args.task)
            value, gradients = value_and_grad(model, *inputs, random_key(meta["seed"], optimizer.step))
            if not np.isfinite(float(value)) or not all(
                bool(mx.all(mx.isfinite(g))) for _, g in tree_flatten(gradients)
            ):
                raise FloatingPointError("Non-finite property loss or gradient")
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.m, optimizer.v)
            meta["cursor"] += len(chosen)
            if optimizer.step % 50 == 0:
                save(args.output / "checkpoint.npz", model, optimizer, meta)
        if meta["cursor"] == len(indices):
            ids = meta["partitions"]["validation"]
            total = 0.0
            for start in range(0, len(ids), meta["batch_size"]):
                chosen = ids[start : start + meta["batch_size"]]
                total += float(loss(model, *batch(args.output, meta["records"], chosen, args.task))) * len(chosen)
            validation = total / len(ids)
            if not np.isfinite(validation):
                raise FloatingPointError("Non-finite validation loss")
            meta.update(epoch=meta["epoch"] + 1, cursor=0, validation=validation)
            if meta["best_validation"] is None or validation < meta["best_validation"]:
                meta.update(best_validation=validation, stale=0)
                save(args.output / "best.npz", model, optimizer, meta)
            else:
                meta["stale"] += 1
            print(json.dumps(dict(epoch=meta["epoch"], validation=validation, step=optimizer.step)), flush=True)
        save(args.output / "checkpoint.npz", model, optimizer, meta)
    report = {k: meta[k] for k in ("epoch", "cursor", "validation", "best_validation", "adaptations") if k in meta}
    report.update(stopped=stopping, early_stopped=meta["stale"] >= meta["patience"], complete_paper_reproduction=False)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
