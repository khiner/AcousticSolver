"""Evaluate matched audio predictions and Appendix D robot source properties."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.signal import hilbert

from . import data, dsp
from .cdpam import CDPAM
from .properties import MATERIALS, features
from .train import load


def read_array(root, item):
    path = (root / item["path"]).resolve()
    if data.digest(path) != item["sha256"]:
        raise ValueError(f"Evaluation input hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        value = archive[item["key"]].astype(np.float32)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError("Expected finite nonempty [recordings, samples] audio")
    return value


def distances(target, prediction, perceptual):
    if target.shape != prediction.shape:
        raise ValueError("Predictions must preserve target order and shape; no implicit alignment")
    result = {"spectral": [], "envelope": [], "cdpam": []}
    for a, b in zip(target, prediction):
        result["spectral"].append(
            float(dsp.spectral_loss(mx.array(a[None]), mx.array(b[None]), (2048, 1024, 512, 256, 128, 64)))
        )
        result["envelope"].append(float(np.sqrt(np.mean((np.abs(hilbert(a)) - np.abs(hilbert(b))) ** 2))))
        result["cdpam"].append(float(perceptual(a[None], b[None])[0]))
    return {k: np.array(v) for k, v in result.items()}


def audio_evaluation(spec, root, weights):
    if spec["rate"] != 44100:
        raise ValueError("Audio evaluation expects 44100 Hz recordings")
    target = read_array(root, spec["target"])
    metric = CDPAM(weights)
    seed = spec.get("noise_seed", 714)
    noise = np.random.default_rng(seed).uniform(-1, 1, target.shape).astype(np.float32)
    denominator = distances(target, noise, metric)
    means = {k: float(v.mean()) for k, v in denominator.items()}
    if any(v <= 0 or not np.isfinite(v) for v in means.values()):
        raise ValueError("White-noise metric denominator is zero or non-finite")
    results = {}
    for name, item in spec["predictions"].items():
        values = distances(target, read_array(root, item), metric)
        if not all(np.isfinite(v).all() for v in values.values()):
            raise FloatingPointError("Non-finite evaluation metric")
        results[name] = dict(
            per_recording={k: v.tolist() for k, v in values.items()},
            mean={k: float(v.mean()) for k, v in values.items()},
            normalized={k: float(v.mean()) / means[k] for k, v in values.items()},
        )
    return dict(
        predictions=results,
        white_noise_mean=means,
        noise_seed=seed,
        adaptations=[
            "uniform white noise in [-1,1]; the paper does not specify its distribution or level",
            "CDPAM uses scipy polyphase 44.1 to 22.05 kHz resampling instead of librosa",
            "no implicit peak normalization, time alignment or trimming",
        ],
        normalization="ratio of dataset means, not mean of per-recording ratios",
    )


def property_predictions(model, audio):
    outputs = []
    for row in audio:
        x = features(row, model._config)
        outputs.append(np.asarray(model(mx.array(x[None]), mx.array([len(x)])))[0])
    values = np.array(outputs)
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite property prediction")
    return values


def property_scores(classes, youngs, labels, target_youngs):
    confusion = np.zeros((4, 4), dtype=int)
    predicted = np.argmax(classes, axis=-1)
    np.add.at(confusion, (labels, predicted), 1)
    errors = 100 * np.abs(youngs - target_youngs) / target_youngs
    return dict(
        accuracy=float(np.mean(predicted == labels)),
        confusion_true_rows=confusion.tolist(),
        mean_absolute_percentage_error=float(errors.mean()),
        percentage_errors=errors.tolist(),
        predicted_youngs_pa=youngs.tolist(),
    )


def assignment_scores(inferred, labels, moduli):
    correct = np.stack([np.argmax(v["material"], axis=-1) == labels for v in inferred.values()])
    errors = np.stack([100 * np.abs(v["youngs"] - moduli) / moduli for v in inferred.values()])
    return dict(
        dataset_max_accuracy=float(correct.mean(axis=1).max()),
        instance_max_accuracy=float(correct.max(axis=0).mean()),
        dataset_min_percentage_error=float(errors.mean(axis=1).min()),
        instance_min_percentage_error=float(errors.min(axis=0).mean()),
    )


def robot_evaluation(spec, root):
    classifiers = {}
    for task in ("material", "youngs"):
        item = spec["checkpoints"][task]
        path = (root / item["path"]).resolve()
        if data.digest(path) != item["sha256"]:
            raise ValueError("Property checkpoint hash mismatch")
        model, _, meta = load(path)
        if meta["experiment"] != "properties" or model._config["task"] != task:
            raise ValueError("Wrong property checkpoint task")
        if meta["step"] == 0 or not meta.get("records"):
            raise ValueError("Evaluation requires trained Sound-20K checkpoints")
        classifiers[task] = model
    targets = spec["targets"]
    raw = read_array(root, spec["original"])
    if len(targets) != len(raw):
        raise ValueError("Need object/tool material and modulus labels for every recording")
    labels, moduli = {}, {}
    for side in ("object", "tool"):
        labels[side] = np.array([MATERIALS.index(t[side]["material"]) for t in targets])
        moduli[side] = np.array([t[side]["youngs_pa"] for t in targets])
        if not np.isfinite(moduli[side]).all() or np.any(moduli[side] <= 0):
            raise ValueError("Ground truth modulus must be positive finite Pascals")
    results = {}
    candidates = dict(spec.get("predictions", {}), original={"object": spec["original"], "tool": spec["original"]})
    for name, candidate in candidates.items():
        signals = {side: read_array(root, candidate[side]) for side in ("object", "tool")}
        if any(a.shape != raw.shape for a in signals.values()):
            raise ValueError("Separated predictions must preserve recording order and shape")
        inferred = {
            side: {task: property_predictions(model, a) for task, model in classifiers.items()}
            for side, a in signals.items()
        }
        scores = {}
        for side in signals:
            values = inferred[side]
            scores[side] = property_scores(values["material"], values["youngs"], labels[side], moduli[side])
            scores[side].update(assignment_scores(inferred, labels[side], moduli[side]))
        results[name] = scores
    return dict(
        material_order=list(MATERIALS),
        predictions=results,
        assignment="Report named-stem confusion and accuracy plus Tables 1/2 dataset and instance extrema for every method",
        training="Synthetic Sound-20K only; real robot recordings are evaluation-only",
    )


def robot_manifest(run, material, youngs):
    metadata = json.loads((run / "run.json").read_text())
    results = json.loads((run / "results.json").read_text())
    if metadata["experiment"] != "separation" or data.digest(run / "dataset.npz") != metadata["dataset_sha256"]:
        raise ValueError("Expected a complete source-separation run with its retained dataset")
    if data.digest(run / "rendered.npz") != results["rendered_sha256"]:
        raise ValueError("Retained separation render hash mismatch")
    with np.load(youngs, allow_pickle=False) as archive:
        records = json.loads(str(archive["metadata"]))["records"]
    moduli = {}
    for record in records:
        label = record["material"]
        if label in moduli and moduli[label] != record["youngs_pa"]:
            raise ValueError("Material has multiple nominal moduli; use explicit evaluation targets")
        moduli[label] = record["youngs_pa"]

    def item(path, **fields):
        return dict(path=str(path.resolve()), sha256=data.digest(path), **fields)

    targets = []
    for source in metadata["sources"]:
        labels = {side: source[side].split("-")[0] for side in ("object", "tool")}
        labels = {side: "polycarb" if label == "polycarbonate" else label for side, label in labels.items()}
        targets.append({side: dict(material=label, youngs_pa=moduli[label]) for side, label in labels.items()})
    return dict(
        checkpoints=dict(material=item(material), youngs=item(youngs)),
        targets=targets,
        original=item(run / "dataset.npz", key="train.audio"),
        predictions={
            "DiffImpact": dict(
                object=item(run / "rendered.npz", key="train.body"),
                tool=item(run / "rendered.npz", key="train.separated_tool"),
            )
        },
        run_metadata_sha256=data.digest(run / "run.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("audio", "robot"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--run", type=Path, help="Build a robot evaluation manifest from a retained separation run")
    parser.add_argument("--material", type=Path, help="Best material-classifier checkpoint for --run")
    parser.add_argument("--youngs", type=Path, help="Best modulus-regressor checkpoint for --run")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cdpam-weights", type=Path, default=Path("build/reference/cdpam/weights.npz"))
    args = parser.parse_args()
    if args.run:
        if args.experiment != "robot" or not args.material or not args.youngs:
            parser.error("--run requires robot evaluation and both property checkpoints")
        spec, root = robot_manifest(args.run, args.material, args.youngs), Path(".")
    else:
        spec, root = json.loads(args.manifest.read_text()), args.manifest.parent
    results = (
        audio_evaluation(spec, root, args.cdpam_weights) if args.experiment == "audio" else robot_evaluation(spec, root)
    )
    results.update(
        manifest=spec,
        complete_paper_reproduction=False,
        implementation={
            name: data.digest(Path(__file__).with_name(name))
            for name in ("evaluate.py", "properties.py", "cdpam.py", "dsp.py", "protocol.json")
        },
    )
    if args.manifest:
        results["manifest_sha256"] = data.digest(args.manifest)
    if args.experiment == "audio":
        results["cdpam_weights_sha256"] = data.digest(args.cdpam_weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
