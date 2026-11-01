"""Check the accepted local numerical baseline without TensorFlow or a remote machine."""

import argparse
import json
from pathlib import Path
import tempfile

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from scipy.special import expit

from . import data, dsp, models
from .train import Adam, load, predictions, random_key, save, step, validate


def function(case):
    name, kwargs = case["name"], case["kwargs"]
    if name == "gru":

        def recurrent(x, w, h, b):
            layer = models.GRU(w.shape[0], h.shape[0])
            layer.Wx, layer.Wh, layer.bias = w.T, h.T, b
            return layer(x)

        return recurrent
    if name == "encoder":

        def encoder(features, *parameters):
            model = models.Encoder(kwargs["channels"])
            model.update(tree_unflatten(list(zip(kwargs["names"], parameters))))
            return model(features, kwargs["frames"])

        return encoder
    if name in ("autoencoder_graph", "autoencoder_loss"):

        def autoencoder(audio, target, *args):
            model = models.Autoencoder(kwargs["config"])
            model.update(tree_unflatten(list(zip(kwargs["names"], args))))
            batch = dict(mfcc=models.mfcc(audio, kwargs["config"]["rate"]))
            output = model(batch, dict(background=args[-2], reverb=args[-1]))
            c = kwargs["config"]
            if name == "autoencoder_loss":
                return dsp.spectral_loss(target, output["audio"], c["fft_sizes"], c["log_weight"])
            controls = model.encoder(batch["mfcc"], c["frames"])
            return mx.concatenate(
                [
                    v.reshape(-1)
                    for v in (
                        controls,
                        output["force"],
                        *(output[k] for k in ("audio", "body", "acceleration", "noise", "reverb")),
                    )
                ]
            )

        return autoencoder
    if name.startswith("property_"):
        from .properties import PropertyModel

        def property_model(features, *parameters):
            model = PropertyModel(kwargs["config"])
            model.update(tree_unflatten(list(zip(kwargs["names"], parameters))))
            return model(features, mx.array(kwargs["lengths"]))

        return property_model
    if name in ("hammer_graph", "separation_graph"):

        def graph(*args):
            model = (models.Hammer if name == "hammer_graph" else models.Separation)(kwargs["config"])
            inputs = dict(zip(kwargs["names"], args))
            draws = {key: inputs.pop(key) for key in ("contact", "background", "reverb") if key in inputs}
            model.update(inputs)
            batch = (
                {}
                if name == "hammer_graph"
                else dict(indices=mx.arange(4), objects=mx.array(kwargs["objects"]), tools=mx.array(kwargs["tools"]))
            )
            return model(batch, draws)["audio"]

        return graph
    operation = getattr(dsp, name, None) or getattr(models, name)
    return lambda *args: operation(*args, **kwargs)


def evaluate(case, fixture):
    name = case["name"]
    args = [mx.array(fixture[f"{name}.input.{i}"]) for i in range(case["inputs"])]
    operation = function(case)
    output = operation(*args)
    cotangent = mx.array(fixture[f"{name}.cotangent"])
    gradient = mx.grad(lambda *x: mx.sum(operation(*x) * cotangent), argnums=tuple(range(len(args))))(*args)
    gradient = (gradient,) if len(args) == 1 else gradient
    return {name + ".output": np.asarray(output), **{f"{name}.grad.{i}": np.asarray(g) for i, g in enumerate(gradient)}}


def check_values(actuals, fixture, tolerance=1e-5):
    maximum = 0
    for key, actual in actuals.items():
        expected = fixture[key]
        error = np.linalg.norm((actual - expected).ravel())
        scale = np.linalg.norm(expected.ravel())
        if not np.all(np.isfinite(actual)) or error > tolerance * scale + 1e-7 * np.sqrt(actual.size):
            raise AssertionError(f"{key}: relative error {error / max(scale, 1e-30):.6g}")
        maximum = max(maximum, float(error / max(scale, 1e-30)))
    return maximum


def compare(fixture, tolerance=1e-5):
    metadata = json.loads(str(fixture["metadata"]))
    maximum = max(check_values(evaluate(case, fixture), fixture, tolerance) for case in metadata["cases"])
    for training in metadata["training"]:
        maximum = max(
            maximum, check_values(evaluate_training(fixture, training), fixture, training["regression_tolerance"])
        )
    model = nn.Module()
    model["value"] = mx.array(fixture["adam_initial"])
    optimizer = Adam()
    for gradient in fixture["adam_gradients"]:
        optimizer.update(model, {"value": mx.array(gradient)})
    np.testing.assert_allclose(np.asarray(model["value"]), fixture["adam_result"], atol=1e-6, rtol=1e-6)
    return dict(cases=len(metadata["cases"]), largest_relative_error=maximum)


def evaluate_training(fixture, training):
    metadata = json.loads(str(fixture["metadata"]))
    case = next(c for c in metadata["cases"] if c["name"] == training["case"])
    args = [mx.array(fixture[f'{case["name"]}.input.{i}']) for i in range(case["inputs"])]
    names = case["kwargs"]["names"]
    encoder = case["name"] == "encoder"
    model = models.Encoder(case["kwargs"]["channels"]) if encoder else models.Autoencoder(case["kwargs"]["config"])
    optimizer = Adam()
    offset = 1 if encoder else 2
    assert set(names) == set(dict(tree_flatten(model.parameters())))
    model.update(tree_unflatten(list(zip(names, args[offset:]))))
    operation = function(case)

    def objective(model):
        parameters = dict(tree_flatten(model.parameters()))
        result = operation(*args[:offset], *(parameters[k] for k in names), *([] if encoder else args[-2:]))
        return mx.sum(result * mx.array(fixture["encoder.cotangent"])) if encoder else result

    values = {}
    for i in range(training["steps"]):
        loss, gradients = nn.value_and_grad(model, objective)(model)
        prefix = case["name"] + ".training"
        values[f"{prefix}.loss.{i}"] = np.asarray(loss)
        optimizer.update(model, gradients)
        for kind, tensors in (
            ("gradient", dict(tree_flatten(gradients))),
            ("parameter", dict(tree_flatten(model.parameters()))),
            ("m", optimizer.m),
            ("v", optimizer.v),
        ):
            for name, tensor in tensors.items():
                values[f"{prefix}.{i}.{kind}.{name}"] = np.asarray(tensor)
    return values


def contracts():
    mx.random.seed(917)
    mags = mx.random.uniform(0.01, 1, (2, 20, 1))
    taus = mx.random.uniform(0.001, 0.003, (2, 20, 1))
    batched = dsp.contact(mags, taus, 800, 8000)
    independent = mx.concatenate([dsp.contact(mags[i : i + 1], taus[i : i + 1], 800, 8000) for i in range(2)])
    np.testing.assert_allclose(np.asarray(batched), np.asarray(independent), atol=1e-6)
    # Constant envelopes have a deterministic earliest-maximum convention.
    flat = dsp.contact(mx.ones((2, 20, 1)), taus, 800, 8000)
    assert bool(mx.all(mx.isfinite(flat)))
    g, f, d = mx.array([[[0.0, 0.5, -1.0]]]), mx.array([[[0.0, 0.1, -0.2]]]), mx.array([[[-5.0, -4.0, -3.0]]])
    gains, frequencies, dampings = [np.asarray(x).astype(np.float64) for x in dsp.modal_controls(g, f, d, 20000)]
    n = np.arange(88200)[None, :, None]
    expected = np.sum(
        gains * np.exp(-dampings * n / 88200) * np.sin((n + 1) * frequencies * 2 * np.pi / 88200), axis=-1
    )
    actual = np.asarray(dsp.modal_ir(g, f, d, 88200, 88200, 20000))[:, 88200:]
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-4
    cotangent = np.random.default_rng(11).normal(size=(1, 88200)).astype(np.float32)
    gradients = mx.grad(
        lambda g, f, d: mx.sum(dsp.modal_ir(g, f, d, 88200, 88200, 20000)[:, 88200:] * mx.array(cotangent)),
        argnums=(0, 1, 2),
    )(g, f, d)
    time = n / 88200
    phase = (n + 1) * frequencies * 2 * np.pi / 88200
    envelope = np.exp(-dampings * time)
    adjoint = cotangent[..., None]
    dg = np.sum(adjoint * envelope * np.sin(phase), axis=1, keepdims=True)
    df = np.sum(adjoint * gains * envelope * np.cos(phase) * (n + 1) * 2 * np.pi / 88200, axis=1, keepdims=True)
    dd = np.sum(-adjoint * gains * time * envelope * np.sin(phase), axis=1, keepdims=True)
    centers = 700 * (10 ** (np.linspace(2595 * np.log10(1 + 10 / 700), 2595 * np.log10(1 + 20000 / 700), 3) / 2595) - 1)
    raw_f = centers + (0.108 * centers + 24.7) * np.tanh(np.asarray(f))
    df *= (expit(raw_f) - expit(raw_f - 19990)) * (0.108 * centers + 24.7) * (1 - np.tanh(np.asarray(f)) ** 2)
    sigmoid = expit(np.asarray(d).astype(float) - 1.5)
    dd *= 20000 * np.log(4) * sigmoid ** np.log(4) * (1 - sigmoid)
    dg = gains * (dg - np.sum(dg * gains, axis=-1, keepdims=True))
    for actual, expected in zip(gradients, (dg, df, dd)):
        assert np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected) < 2e-3
    config = dict(
        rate=4000,
        samples=400,
        frames=20,
        bodies=2,
        modes=8,
        ir_samples=128,
        hz_max=3500,
        reverb_samples=1000,
        fft_sizes=[256, 128],
        log_weight=1.0,
    )
    target = mx.random.normal((2, 400)) * 0.1
    with tempfile.TemporaryDirectory() as directory:
        for name, cls in models.MODELS.items():
            c = dict(config)
            batch = {"audio": target}
            if name == "hammer":
                c.update(bodies=1, recordings=2)
            if name == "autoencoder":
                c.update(bodies=1, channels=16)
                batch["mfcc"] = models.mfcc(target, 8000)
            if name == "separation":
                c.update(tools=2, recordings=2, peak_times=[2.0, 2.0])
                batch.update(indices=mx.arange(2), tools=mx.array([0, 1]), objects=mx.array([0, 1]))
            model, optimizer = cls(c), Adam()
            frozen = ["gains", "frequencies", "dampings"] if name == "hammer" else []
            model.freeze(keys=frozen, strict=True)
            metadata = dict(config=c, experiment=name, frozen_parameters=frozen)
            frozen_values = {key: np.asarray(model[key]).copy() for key in frozen}
            step(model, optimizer, batch, 5)
            path = Path(directory) / "checkpoint.npz"
            save(path, model, optimizer, metadata)
            restored, resumed, _ = load(path)
            for original, recovered in (
                (model.parameters(), restored.parameters()),
                (optimizer.m, resumed.m),
                (optimizer.v, resumed.v),
            ):
                for (parameter, a), (other, b) in zip(tree_flatten(original), tree_flatten(recovered)):
                    assert parameter == other
                    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            first, second = step(model, optimizer, batch, 5), step(restored, resumed, batch, 5)
            np.testing.assert_allclose(first, second, rtol=1e-6)
            for (_, a), (_, b) in zip(tree_flatten(model.parameters()), tree_flatten(restored.parameters())):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=2e-6, rtol=1e-6)
            for key, value in frozen_values.items():
                np.testing.assert_array_equal(np.asarray(restored[key]), value)
            rendered = model(batch, model.noise(2, random_key(5, 0)))
            components = ["body", "acceleration", "noise", "reverb"] + (["tool"] if name == "separation" else [])
            np.testing.assert_allclose(
                np.asarray(rendered["audio"]), np.asarray(sum(rendered[k] for k in components)), atol=2e-6, rtol=1e-5
            )


def shared_body_contract():
    config = dict(
        rate=4000,
        samples=400,
        frames=20,
        bodies=1,
        recordings=3,
        modes=8,
        ir_samples=128,
        hz_max=3500,
        reverb_samples=1000,
    )
    mx.random.seed(932)
    shared = models.Hammer(config)
    independent = models.Hammer(dict(config, bodies=3))
    body_keys = ("gains", "frequencies", "dampings")
    independent.update({k: mx.repeat(v, 3, axis=0) if k in body_keys else v for k, v in shared.parameters().items()})
    draws = shared.noise(3, random_key(719, 0))
    expected_output = independent({}, draws)
    for key, value in shared({}, draws).items():
        np.testing.assert_allclose(np.asarray(value), np.asarray(expected_output[key]), atol=1e-6, rtol=1e-5)
    cotangent = mx.random.normal((3, 400))
    loss = lambda model: mx.sum(model({}, draws)["audio"] * cotangent)
    _, shared_grad = nn.value_and_grad(shared, loss)(shared)
    _, independent_grad = nn.value_and_grad(independent, loss)(independent)
    for key, gradient in shared_grad.items():
        expected = mx.sum(independent_grad[key], axis=0, keepdims=True) if key in body_keys else independent_grad[key]
        np.testing.assert_allclose(np.asarray(gradient), np.asarray(expected), atol=2e-6, rtol=1e-5, err_msg=key)


def render_signature(output):
    result = {key: np.asarray(value)[:, ::64] for key, value in output.items() if key != "force"}
    force = np.asarray(output["force"])
    force = np.pad(force, ((0, 0), (0, (-force.shape[1]) % 128)))
    result["force"] = force.reshape(force.shape[0], -1, 128).sum(axis=-1)
    return result


def retained(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, info in manifest["fixtures"].items():
        if data.digest(root / name) != info["sha256"]:
            raise ValueError(f"Retained fixture changed: {name}")
    with np.load(root / "rendered.npz", allow_pickle=False) as fixture:
        for name in ("hammer", "separation"):
            model, _, metadata = load(root / f"{name}.npz")
            count = model._config.get("recordings", model._config["bodies"])
            batch = (
                {key: mx.array(fixture[f"{name}.input.{key}"]) for key in ("indices", "objects", "tools")}
                if name == "separation"
                else {}
            )
            output = model(batch, model.noise(count, random_key(714, 0)))
            for key, value in render_signature(output).items():
                np.testing.assert_allclose(
                    value, fixture[f"{name}.{key}"], atol=1e-6, rtol=2e-5, err_msg=f"{name} trained state: {key}"
                )


def property_contracts():
    from .evaluate import assignment_scores
    from .properties import CONFIG, MATERIALS, PropertyModel, loss, split

    rng = np.random.default_rng(4)
    for task in ("material", "youngs"):
        mx.random.seed(5)
        model = PropertyModel(dict(CONFIG, task=task, channels=8, hidden=4, window=16))
        x = mx.array(rng.uniform(0.1, 1, (2, 13, 9)).astype(np.float32))
        lengths = mx.array([13, 9])
        labels = mx.array([0, 2]) if task == "material" else mx.array([7.4e10, 2e11])
        predictions = model(x, lengths)
        np.testing.assert_allclose(
            np.asarray(predictions)[1], np.asarray(model(x[1:2, :9], mx.array([9])))[0], rtol=1e-6
        )
        # The final bias has a closed-form derivative for each supervised objective.
        _, gradients = nn.value_and_grad(model, loss)(model, x, lengths, labels)
        if task == "material":
            expected = mx.mean(mx.softmax(predictions, axis=-1) - mx.eye(4)[labels], axis=0)
        else:
            expected = mx.mean(2 * (mx.log1p(predictions) - mx.log1p(labels)) * predictions / (1 + predictions))
        np.testing.assert_allclose(np.asarray(gradients["head"]["bias"]), np.asarray(expected), rtol=2e-6, atol=1e-7)
        optimizer = Adam()

        def update(model, optimizer):
            value, gradient = nn.value_and_grad(model, loss)(model, x, lengths, labels, random_key(3, optimizer.step))
            optimizer.update(model, gradient)
            mx.eval(model.parameters(), optimizer.m, optimizer.v)
            assert np.isfinite(float(value))
            return float(value)

        update(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save(path, model, optimizer, dict(experiment="properties", config=model._config))
            restored, resumed, _ = load(path)
            np.testing.assert_equal(update(model, optimizer), update(restored, resumed))
            for (_, a), (_, b) in zip(tree_flatten(model.parameters()), tree_flatten(restored.parameters())):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)
    records = [
        dict(material=material, group=f"{material}-{i // 2}") for material in (*MATERIALS, "bronze") for i in range(40)
    ]
    for task in ("material", "youngs"):
        partitions = split(records, task, 714)
        a, b = ({records[i]["group"] for i in partitions[k]} for k in ("train", "validation"))
        assert not a & b and a and b
        if task == "youngs":
            assert sum(map(len, partitions.values())) == len(records)
        else:
            for ids in partitions.values():
                assert len({sum(records[i]["material"] == m for i in ids) for m in MATERIALS}) == 1
    # Per-instance extrema must not be confused with selecting one source for the dataset.
    inferred = {
        str(i): dict(material=np.eye(4)[[i, i]], youngs=np.array([1.0, 3.0]) if i == 0 else np.array([3.0, 1.0]))
        for i in range(2)
    }
    scores = assignment_scores(inferred, np.array([0, 1]), np.ones(2))
    assert scores == dict(
        dataset_max_accuracy=0.5,
        instance_max_accuracy=1.0,
        dataset_min_percentage_error=100.0,
        instance_min_percentage_error=0.0,
    )


def metric_contracts(root, weights):
    from .cdpam import CDPAM

    metric = CDPAM(weights)
    with np.load(root / "cdpam.npz", allow_pickle=False) as fixture:
        a, b = metric.embedding(fixture["a"]), metric.embedding(fixture["b"])
        for key, actual in (("embedding_a", a), ("embedding_b", b), ("distance", metric.distance(a, b))):
            np.testing.assert_allclose(np.asarray(actual), fixture[key], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(metric.distance(a, a)), 0, atol=1e-7)
        np.testing.assert_allclose(np.asarray(metric.distance(a, b)), np.asarray(metric.distance(b, a)), atol=1e-7)


def selection_contract():
    config = dict(
        rate=4000,
        samples=400,
        frames=20,
        bodies=1,
        channels=16,
        modes=8,
        ir_samples=128,
        hz_max=3500,
        reverb_samples=1000,
        fft_sizes=[256, 128],
        log_weight=1.0,
    )
    mx.random.seed(123)
    model, optimizer = models.Autoencoder(config), Adam()
    batch = dict(audio=mx.zeros((2, 400)), mfcc=mx.random.normal((2, 4, 40)))
    batch["audio"] = model(batch, model.noise(2, random_key(714, 0)))["audio"]
    metadata = dict(experiment="autoencoder", config=config, batch_size=2)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        validate(root, model, optimizer, metadata, batch)
        checksum = data.digest(root / "best.npz")
        optimizer.step = 1
        model.acceleration_scale = model.acceleration_scale + 1000
        validate(root, model, optimizer, metadata, batch)
        validate(root, model, optimizer, metadata, batch)
        assert metadata["best_step"] == 0 and metadata["best_validation"] == 0
        assert len(metadata["validation_history"]) == 2 and metadata["validation_history"][1]["loss"] > 0
        assert data.digest(root / "best.npz") == checksum
        restored, state, _ = load(root / "best.npz")
        assert state.step == 0 and next(predictions(restored, batch, 2))[1] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cdpam-weights", type=Path, help="Also check the frozen perceptual metric against its local baseline"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "tests/fixtures/diffimpact"
    with np.load(root / "baseline.npz", allow_pickle=False) as fixture:
        report = compare(fixture)
    contracts()
    shared_body_contract()
    retained(root)
    property_contracts()
    selection_contract()
    if args.cdpam_weights:
        metric_contracts(root, args.cdpam_weights)
    print(
        json.dumps(
            dict(
                report,
                contracts=True,
                shared_body=True,
                trained_states=True,
                property_contracts=True,
                validation_selection=True,
                cdpam=bool(args.cdpam_weights),
            )
        )
    )
