"""One-time comparison using the released encoder/decoder classes and explicit ASMR wiring."""

from unittest.mock import patch
from pathlib import Path
import sys
import types

import numpy as np


def capture(tf, ddsp, normal, record, values):
    # Load the unchanged layers without the training package's unrelated cloud integrations.
    package = types.ModuleType("ddsp.training")
    package.__path__ = [str(Path(ddsp.__file__).parent / "training")]
    sys.modules["ddsp.training"] = package
    from ddsp.training import encoders, decoders

    config = dict(
        rate=44100,
        samples=3000,
        frames=20,
        bodies=1,
        channels=16,
        modes=8,
        ir_samples=128,
        hz_max=20000,
        reverb_samples=1000,
        fft_sizes=[256, 128],
        log_weight=1.0,
    )
    encoder = encoders.MfccTimeDistributedRnnEncoder(
        rnn_channels=config["channels"], z_time_steps=config["frames"], sample_rate=config["rate"]
    )
    decoder = decoders.RnnFcDecoder(
        rnn_channels=config["channels"],
        ch=config["channels"],
        input_keys=("z",),
        output_splits=(("magnitudes", 1), ("taus", 1)),
    )
    audio, target = normal((2, config["samples"])), normal((2, config["samples"]))
    decoder(encoder(tf.constant(audio)))
    bindings = []

    def bind(name, layer, attribute, axes=None, shape=None):
        variable = getattr(layer, attribute)
        value = normal(variable.shape)
        if attribute in ("scale", "gamma"):
            value += 1
        if axes is not None:
            value = value.transpose(np.argsort(axes))
        if shape:
            value = value.reshape(shape)
        bindings.append((name, layer, attribute, axes, tuple(variable.shape)))
        return value

    parameters = [
        bind("scale", encoder.z_norm, "scale", shape=(1, 1, 40)),
        bind("shift", encoder.z_norm, "shift", shape=(1, 1, 40)),
    ]
    for name, layer in (("input_gru", encoder.rnn.rnn), ("output_gru", decoder.rnn.rnn)):
        parameters.extend(
            [
                bind(name + ".Wx", layer.cell, "kernel", (1, 0)),
                bind(name + ".Wh", layer.cell, "recurrent_kernel", (1, 0)),
                bind(name + ".bias", layer.cell, "bias"),
            ]
        )
    parameters.extend([bind("conv.weight", encoder.tcnn, "kernel", (1, 2, 0)), bind("conv.bias", encoder.tcnn, "bias")])
    for name, layer in (("projection", encoder.dense_out), ("head", decoder.dense_out)):
        parameters.extend([bind(name + ".weight", layer, "kernel", (1, 0)), bind(name + ".bias", layer, "bias")])
    for name, stack in (("input_fc", decoder.input_stacks[0]), ("output_fc", decoder.out_stack)):
        for i, block in enumerate(stack.layers):
            dense, norm = block.layers[:2]
            parameters.extend(
                [
                    bind(f"{name}.layers.{3*i}.weight", dense, "kernel", (1, 0)),
                    bind(f"{name}.layers.{3*i}.bias", dense, "bias"),
                    bind(f"{name}.layers.{3*i+1}.weight", norm, "gamma"),
                    bind(f"{name}.layers.{3*i+1}.bias", norm, "beta"),
                ]
            )
    assert {id(getattr(layer, attribute)) for _, layer, attribute, _, _ in bindings} == {
        id(v) for v in encoder.trainable_variables + decoder.trainable_variables
    }

    def predict(x, weights):
        for (_, layer, attribute, axes, shape), value in zip(bindings, weights):
            setattr(layer, attribute, tf.transpose(value, axes) if axes else tf.reshape(value, shape))
        output = decoder(encoder(x))
        return tf.concat((output["magnitudes"], output["taus"]), axis=-1)

    def from_features(features, *weights):
        with patch.object(ddsp.spectral_ops, "compute_mfcc", return_value=features):
            return predict(tf.zeros_like(audio), weights)

    features = normal((2, 12, 40))
    record(
        "encoder",
        from_features,
        [features, *parameters],
        dict(channels=config["channels"], frames=config["frames"], names=[b[0] for b in bindings]),
    )

    def train(name, objective, names, initial):
        variables = [tf.Variable(p) for p in initial]
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, clipnorm=1.0)
        for i in range(3):
            with tf.GradientTape() as tape:
                value = objective(*variables)
            gradients = tape.gradient(value, variables)
            values[f"{name}.training.loss.{i}"] = value.numpy()
            optimizer.apply_gradients(zip(gradients, variables))
            for key, variable, gradient in zip(names, variables, gradients):
                if key == "acceleration_scale":
                    variable.assign(tf.maximum(variable, 0))
                for kind, tensor in (
                    ("gradient", gradient),
                    ("parameter", variable),
                    ("m", optimizer.get_slot(variable, "m")),
                    ("v", optimizer.get_slot(variable, "v")),
                ):
                    values[f"{name}.training.{i}.{kind}.{key}"] = tensor.numpy()
        return dict(case=name, steps=3, regression_tolerance=5e-4 if name == "autoencoder_loss" else 1e-5)

    training = [
        train(
            "encoder",
            lambda *p: tf.reduce_sum(from_features(tf.constant(features), *p) * values["encoder.cotangent"]),
            [b[0] for b in bindings],
            parameters,
        )
    ]
    names = ["encoder." + b[0] for b in bindings]
    # Keep noise and reverb audible to exercise their gradients away from spectral zeros.
    physical = dict(
        gains=normal((1, 1, 8)),
        frequencies=normal((1, 1, 8)),
        dampings=normal((1, 1, 8)),
        noise_magnitudes=normal((1, 200)) + 3,
        acceleration_scale=np.full((1, 1), 0.2, np.float32),
        reverb_gains=normal((1, 1, 32)) + 3,
        reverb_decay=normal((1, 1, 32)) - 3,
        tau_bias=normal((1, 1, 1)),
    )
    names += list(physical)
    parameters += list(physical.values())
    noise = [normal((2, config["samples"] * 2)), normal((2, config["reverb_samples"]))]
    impact = ddsp.synths.Impact(
        n_samples=config["samples"] * 2, sample_rate=config["rate"] * 2, max_impact_frequency=20, include_noise=False
    )
    modal = ddsp.synths.ModalFIR(
        n_samples=config["ir_samples"], sample_rate=config["rate"] * 2, hz_max=config["hz_max"], freq_scale="mel"
    )
    reverb = ddsp.effects.FilteredNoiseExpDecayReverb(reverb_length=config["reverb_samples"], trainable=False)
    loss = ddsp.losses.SpectralLoss(fft_sizes=config["fft_sizes"], logmag_weight=config["log_weight"])

    def render(x, target, *args):
        controls = predict(x, args[: len(bindings)])
        p = dict(zip(names, args))
        scaled = impact.get_controls(
            controls[..., :1], tf.zeros_like(controls[..., :1]), controls[..., 1:], p["tau_bias"]
        )
        # Preserve the explicit batch-index correction, leaving upstream source untouched.
        force = tf.concat([impact.get_signal(**{k: v[i : i + 1] for k, v in scaled.items()}) for i in range(2)], 0)
        ir = modal.get_signal(**modal.get_controls(p["gains"], p["frequencies"], p["dampings"]))
        body = ddsp.core.fft_convolve(force, tf.tile(ir, (2, 1)))
        acceleration = force * p["acceleration_scale"]
        background = ddsp.core.frequency_filter(
            args[-2], tf.tile(ddsp.core.exp_sigmoid(p["noise_magnitudes"] - 4), (2, 1)), 257
        )
        dry = body + acceleration + background
        with patch.object(ddsp.synths.tf.random, "uniform", return_value=args[-1]):
            room = reverb.get_controls(
                dry, tf.tile(p["reverb_gains"], (2, 1, 1)), tf.tile(p["reverb_decay"], (2, 1, 1))
            )["ir"]
        wet = ddsp.core.fft_convolve(dry, reverb._mask_dry_ir(room), delay_compensation=0)
        stems = [
            ddsp.core.resample(v, config["samples"])
            for v in (reverb.get_signal(dry, room), body, acceleration, background, wet)
        ]
        return controls, force, stems, loss.call(target, stems[0])

    def graph(*args):
        controls, force, stems, _ = render(*args)
        return tf.concat([tf.reshape(v, [-1]) for v in (controls, force, *stems)], 0)

    def objective(*args):
        return render(*args)[-1]

    inputs = [audio, target, *parameters, *noise]
    kwargs = dict(config=config, names=names)
    record("autoencoder_graph", graph, inputs, kwargs)
    record("autoencoder_loss", objective, inputs, kwargs)
    training.append(
        train(
            "autoencoder_loss",
            lambda *p: objective(tf.constant(audio), tf.constant(target), *p, *map(tf.constant, noise)),
            names,
            parameters,
        )
    )
    return training
