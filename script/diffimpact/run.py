"""Run the pinned DiffImpact analysis-by-synthesis notebook."""

import argparse
import ast
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .data import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")


def reference(checkout):
    checkout = Path(checkout).resolve()
    cfg = json.loads(PROTOCOL.read_text())
    for name, expected in cfg["sources"].items():
        if digest(checkout / name) != expected:
            raise ValueError(f"DiffImpact reference changed: {name}")
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    sys.path.insert(0, str(checkout / "ddsp"))
    import tensorflow as tf
    import ddsp

    if Path(ddsp.__file__).resolve() != checkout / "ddsp/ddsp/__init__.py":
        raise ValueError("wrong DDSP import")
    names = ["ceramic", "polycarb", "steel", "wood"]
    clips = []
    forces = []
    transforms = []
    for name in names:
        rate, audio = wavfile.read(checkout / f"data/hammer/{name}.wav")
        if rate != 44100 or audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("unexpected author recording")
        audio = audio[int(rate * 0.15) : int(rate * 3.15)]
        force = audio[:, 1]
        clip = audio[:, 0]
        mean = float(np.mean(clip))
        clip = clip - mean
        peak = float(np.max(abs(clip)))
        clips.append(clip / peak)
        forces.append(force)
        transforms.append(
            dict(crop_start=int(rate * 0.15), frames=len(clip), audio_mean=mean, audio_peak_after_centering=peak)
        )
    clips = np.array(clips, dtype=np.float32)
    forces = np.array(forces, dtype=np.float32)
    ns = dict(tf=tf, ddsp=ddsp, np=np, fs=rate, clip=clip, clips=clips)
    notebook = json.loads((checkout / "AnalysisBySynthesis.ipynb").read_text())
    for index in (5, 6, 8):
        tree = ast.parse("".join(notebook["cells"][index]["source"]))
        if index == 8:
            tree.body.pop()  # Use the legacy Adam implementation corresponding to the notebook's TensorFlow 2.4 API.
        exec(compile(tree, f"AnalysisBySynthesis.ipynb:cell{index}", "exec"), ns)
    ns["opt"] = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, clipnorm=1.0)
    keys = ("noise_magnitudes", "frequencies", "dampings", "gains", "taus", "mags", "stdevs", "acceleration_scale")
    ns.update(variables=[ns[k] for k in keys], keys=keys, names=names, forces=forces, transforms=transforms, config=cfg)
    return ns


def save(ns, output, step, report):
    rate = ns["fs"]
    internal = rate * ns["sample_factor"]
    # A single generation supplies the retained prediction and matching stochastic contact profile.
    captured = {}
    original = ns["impact"].get_signal

    def capture(*args, **kwargs):
        captured["force"] = original(*args, **kwargs)
        return captured["force"]

    ns["impact"].get_signal = capture
    try:
        prediction = ns["generate_audio"]().numpy()
    finally:
        ns["impact"].get_signal = original
    controls = ns["modal_fir"].get_controls(ns["gains"], ns["frequencies"], ns["dampings"])
    ir = ns["modal_fir"].get_signal(**controls).numpy()
    n = ir.shape[1] // 2
    parameters = {key: ns[key].numpy() for key in (*ns["keys"], "reverb_gains", "reverb_decay")}
    np.savez(output / "parameters.npz", **parameters)
    for i, name in enumerate(ns["names"]):
        directory = output / name
        directory.mkdir(exist_ok=True)
        values = {k: v.numpy()[i, 0].astype(float) for k, v in controls.items()}
        for filename, audio, sr in [
            ("target", ns["clips"][i], rate),
            ("fitted", prediction[i], rate),
            ("measured-force", ns["forces"][i], rate),
            ("inferred-force", captured["force"].numpy()[i], internal),
            ("body", np.pad(ir[i, n:], (2, 510)), internal),
        ]:
            wavfile.write(directory / (filename + ".wav"), sr, np.asarray(audio, dtype=np.float64))
        write_json(
            directory / "result.json",
            dict(
                step=step,
                parameters={k: v.tolist() for k, v in values.items()},
                transformations=ns["transforms"][i],
                artifacts={p.name: digest(p) for p in directory.iterdir() if p.name != "result.json"},
            ),
        )
    report.update(step=step, parameters_sha256=digest(output / "parameters.npz"))
    write_json(output / "report.json", report)


def run(checkout, output, steps=10000, threads=4):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    ns = reference(checkout)
    tf = ns["tf"]
    cfg = ns["config"]
    started = time.perf_counter()
    report = dict(
        complete=False,
        requested_steps=steps,
        full_notebook_schedule=steps == cfg["steps"],
        protocol=cfg,
        backend="pinned author TensorFlow/DDSP",
        tensorflow=tf.__version__,
        numpy=np.__version__,
        threads=threads,
        environment={
            p: importlib.metadata.version(p)
            for p in ("tensorflow", "tf-keras", "numpy", "scipy", "gin-config", "tensorflow-probability")
        },
        runner_sha256=digest(__file__),
        protocol_sha256=digest(PROTOCOL),
        adaptations=[
            "TensorFlow 2.16.2 / legacy Keras on Apple Silicon instead of TensorFlow 2.4.2 Linux",
            "Headless execution; original optimizer settings and final trainable parameters retained without notebook plotting",
        ],
        history=[],
    )
    write_json(output / "report.json", report)
    for step in range(steps + 1):
        if step % cfg["check_interval"] == 0 or step == steps:
            loss = float(ns["loss_func"]())
            report["history"].append(dict(step=step, loss=loss, seconds=time.perf_counter() - started))
            if not np.isfinite(loss):
                raise RuntimeError("nonfinite author loss")
            write_json(output / "report.json", report)
            print(report["history"][-1], flush=True)
        if step == steps:
            break
        ns["opt"].minimize(ns["loss_func"], var_list=ns["variables"])
    save(ns, output, steps, report)
    report.update(complete=True, seconds=time.perf_counter() - started)
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "external/diffimpact")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.steps <= 10000 or args.threads < 1:
        parser.error("steps must be 0..10000; threads must be positive")
    run(args.reference, args.output, args.steps, args.threads)
