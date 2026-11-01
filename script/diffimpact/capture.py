"""Capture portable, independent DDSP forward/VJP fixtures when adding mathematical coverage."""

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from .data import verify_reference


def capture(checkout, output):
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    verify_reference(checkout)
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    sys.path.insert(0, str(checkout / "ddsp"))
    import tensorflow as tf
    import ddsp
    from ddsp import spectral_ops

    rng = np.random.default_rng(715)
    values, cases = {}, []

    def normal(shape):
        return rng.normal(0, 0.3, shape).astype(np.float32)

    def record(name, function, inputs, kwargs=None):
        args = [tf.Variable(x) for x in inputs]
        with tf.GradientTape() as tape:
            result = function(*args)
            cotangent = normal(result.shape)
            objective = tf.reduce_sum(result * cotangent)
        gradients = tape.gradient(objective, args, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        for i, (x, grad) in enumerate(zip(inputs, gradients)):
            values[f"{name}.input.{i}"] = x
            values[f"{name}.grad.{i}"] = tf.convert_to_tensor(grad).numpy()
        values[f"{name}.output"] = result.numpy()
        values[f"{name}.cotangent"] = cotangent
        cases.append(dict(name=name, inputs=len(inputs), kwargs=kwargs or {}))

    record("exp_sigmoid", ddsp.core.exp_sigmoid, [normal((2, 17))])
    record(
        "resample", lambda x: ddsp.core.resample(x, 800, "window"), [normal((2, 20, 1))], dict(size=800, window=True)
    )
    record("convolve", ddsp.core.fft_convolve, [normal((2, 512)), normal((2, 8, 17))])
    record(
        "filter_ir",
        lambda x: ddsp.core.frequency_impulse_response(x, 31),
        [np.abs(normal((2, 3, 33)))],
        dict(window_size=31),
    )
    record(
        "filtered_noise",
        lambda a, m: ddsp.core.frequency_filter(a, m, 257),
        [normal((2, 500)), np.abs(normal((2, 10, 16)))],
    )
    modal = ddsp.synths.ModalFIR(n_samples=128, sample_rate=88200, hz_max=20000, freq_scale="mel")

    def modal_call(g, f, d):
        controls = modal.get_controls(g, f, d)
        return modal.get_signal(**controls)

    record(
        "modal_ir",
        modal_call,
        [normal((2, 1, 8)), normal((2, 1, 8)), normal((2, 1, 8)) - 1],
        dict(size=128, rate=88200, hz_max=20000),
    )
    impact = ddsp.synths.Impact(n_samples=800, sample_rate=8000, max_impact_frequency=20)
    # Per-recording calls avoid the released flattened-index bug without editing upstream.
    magnitudes = np.abs(normal((2, 20, 1))) + 0.1
    magnitudes[:, -1] = 0.01  # Avoid the held endpoint's nondifferentiable flat maximum.
    record(
        "contact",
        lambda m, t: tf.concat([impact.get_signal(m[i : i + 1], t[i : i + 1]) for i in range(2)], 0),
        [magnitudes, np.abs(normal((2, 20, 1))) * 0.003 + 0.001],
        dict(size=800, rate=8000),
    )
    reverb = ddsp.effects.FilteredNoiseExpDecayReverb(reverb_length=1000, trainable=False)

    def room(g, d, noise):
        with patch.object(ddsp.synths.tf.random, "uniform", return_value=noise):
            return reverb._mask_dry_ir(reverb.get_controls(tf.zeros((2, 512)), g, d)["ir"])

    record("reverb_ir", room, [normal((2, 1, 16)), normal((2, 1, 16)), normal((2, 1000))])
    record("magnitude", lambda x: spectral_ops.compute_mag(x, 256), [normal((2, 600))], dict(size=256))
    loss = ddsp.losses.SpectralLoss(fft_sizes=(512, 256, 128), logmag_weight=1)
    record("spectral_loss", loss.call, [normal((2, 600)), normal((2, 600))], dict(sizes=[512, 256, 128]))
    record(
        "mfcc",
        lambda x: spectral_ops.compute_mfcc(
            x, sample_rate=44100, lo_hz=4, hi_hz=16000, fft_size=1024, mel_bins=128, mfcc_bins=40
        ),
        [normal((2, 3000))],
        dict(rate=44100),
    )
    gru = tf.keras.layers.GRU(5, return_sequences=True)
    gru.build((2, 9, 4))

    def recurrent(x, w, h, b):
        gru.cell.kernel, gru.cell.recurrent_kernel, gru.cell.bias = w, h, b
        return gru(x)

    record("gru", recurrent, [normal((2, 9, 4)), normal((4, 15)), normal((5, 15)), normal((2, 15))])
    config = dict(
        rate=4000, samples=400, frames=20, bodies=2, modes=8, ir_samples=128, hz_max=3500, reverb_samples=1000
    )
    modal = ddsp.synths.ModalFIR(n_samples=128, sample_rate=8000, hz_max=3500, freq_scale="mel")
    names = [
        "mags",
        "taus",
        "stdevs",
        "gains",
        "frequencies",
        "dampings",
        "noise_magnitudes",
        "acceleration_scale",
        "reverb_gains",
        "reverb_decay",
        "contact",
        "background",
        "reverb",
    ]
    inputs = [normal((2, 20, 1)) for _ in range(3)] + [normal((2, 1, 8)) for _ in range(3)]
    inputs += [
        normal((2, 200)),
        normal((2, 1)),
        normal((2, 1, 32)),
        normal((2, 1, 32)),
        normal((2, 20, 1)),
        normal((2, 800)),
        normal((2, 1000)),
    ]
    inputs[0][:, -1] = -5

    def graph(*args):
        p = dict(zip(names, args))
        m = ddsp.core.exp_sigmoid(p["mags"] + tf.abs(p["stdevs"]) * p["contact"] - 1.5)
        t = ddsp.core.exp_sigmoid(p["taus"], exponent=4.0, max_value=0.003, threshold=3 / 8000)
        force = tf.concat([impact.get_signal(m[i : i + 1], t[i : i + 1]) for i in range(2)], 0)
        controls = modal.get_controls(p["gains"], p["frequencies"], p["dampings"])
        audio = ddsp.core.fft_convolve(force, modal.get_signal(**controls))
        audio += force * p["acceleration_scale"]
        audio += ddsp.core.frequency_filter(p["background"], ddsp.core.exp_sigmoid(p["noise_magnitudes"] - 4), 257)
        with patch.object(ddsp.synths.tf.random, "uniform", return_value=p["reverb"]):
            ir = reverb.get_controls(audio, p["reverb_gains"], p["reverb_decay"])["ir"]
        return ddsp.core.resample(reverb.get_signal(audio, ir), 400, "linear")

    record("hammer_graph", graph, inputs, dict(config=config, names=names))
    config = dict(config, tools=2, recordings=4, peak_times=[2.0] * 4)
    names = [
        "mag",
        "tau",
        "peak_time",
        "gains",
        "frequencies",
        "dampings",
        "tool_gains",
        "tool_frequencies",
        "tool_dampings",
        "noise_magnitudes",
        "noise_scale",
        "acceleration_scale",
        "reverb_gains",
        "reverb_decay",
        "background",
        "reverb",
    ]
    inputs = [np.abs(normal((4, 1, 1))), np.abs(normal((4, 1, 1))), np.full((4, 1, 1), 2, np.float32)]
    inputs += [normal((2, 1, n)) for n in (8, 8, 8, 4, 4, 4)]
    inputs += [
        normal((1, 100)),
        np.abs(normal((4, 1))),
        np.abs(normal((4, 1))),
        normal((2, 1, 16)),
        normal((2, 1, 16)),
        normal((4, 800)),
        normal((4, 1000)),
    ]
    objects, tools = [0, 1, 0, 1], [0, 0, 1, 1]

    def separated(*args):
        p = dict(zip(names, args))
        force = tf.reduce_sum(
            p["mag"] * impact.hertz_gaussian(p["peak_time"] * 0.01, p["tau"] * 0.001 + 3 / 8000), axis=-1
        )
        audio = tf.zeros_like(force)
        for prefix, ids in (("", objects), ("tool_", tools)):
            controls = modal.get_controls(p[prefix + "gains"], p[prefix + "frequencies"], p[prefix + "dampings"])
            audio += ddsp.core.fft_convolve(force, tf.gather(modal.get_signal(**controls), ids))
        audio += force * (p["acceleration_scale"] + 1e-7)
        magnitudes = tf.tile(ddsp.core.exp_sigmoid(p["noise_magnitudes"] - 3.5), (4, 1))
        audio += p["noise_scale"] * ddsp.core.frequency_filter(p["background"], magnitudes, 64)
        with patch.object(ddsp.synths.tf.random, "uniform", return_value=p["reverb"]):
            ir = reverb.get_controls(audio, tf.gather(p["reverb_gains"], tools), tf.gather(p["reverb_decay"], tools))[
                "ir"
            ]
        return ddsp.core.resample(reverb.get_signal(audio, ir), 400, "linear")

    record("separation_graph", separated, inputs, dict(config=config, names=names, objects=objects, tools=tools))
    initial = normal((2, 3))
    updates = [normal((2, 3)) * 10 for _ in range(3)]
    parameter = tf.Variable(initial)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, clipnorm=1.0)
    for gradient in updates:
        optimizer.apply_gradients([(tf.constant(gradient), parameter)])
    values.update(adam_initial=initial, adam_gradients=np.array(updates), adam_result=parameter.numpy())
    from .capture_encoder import capture as capture_encoder

    training = capture_encoder(tf, ddsp, normal, record, values)
    values["metadata"] = np.array(
        json.dumps(
            dict(
                cases=cases,
                protocol=protocol,
                tensorflow=tf.__version__,
                training=training,
                contact="Independent calls correct released batch-index bug",
            )
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        np.savez_compressed(file, **values)
    print(f"Captured {len(cases)} forward and vector-Jacobian comparisons: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("external/diffimpact"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.reference.resolve(), args.output)
