"""Infer contact and synthesize new recordings with a trained object-specific audio encoder."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.io import wavfile

from . import data, dsp, models
from .train import load, random_key


def infer(checkpoint, recording, output, channel=0):
    model, _, metadata = load(checkpoint)
    if metadata["experiment"] != "autoencoder":
        raise ValueError("New-recording inference requires a trained audio encoder")
    rate, audio = wavfile.read(recording)
    if rate != model._config["rate"]:
        raise ValueError("Recording sample rate must match the trained model")
    if audio.ndim == 2:
        audio = audio[:, channel]
    if not len(audio):
        raise ValueError("Recording is empty")
    size = model._config["samples"]
    rendered, magnitudes, durations, transforms = [], [], [], []
    for i, start in enumerate(range(0, len(audio), size)):
        original = audio[start : start + size].astype(np.float64)
        mean = original.mean()
        peak = np.max(np.abs(original - mean))
        normalized = data.normalize(original)
        clip = mx.array(np.pad(normalized, (0, size - len(original))).astype(np.float32)[None])
        mags, taus = model.controls(models.mfcc(clip, rate))
        result = model.render_contact(mags, taus, model.noise(1, random_key(714, i)))
        magnitudes.append(np.asarray(mags))
        durations.append(np.asarray(taus))
        rendered.append(
            {key: np.asarray(value)[0, : len(original) * (2 if key == "force" else 1)] for key, value in result.items()}
        )
        transforms.append(dict(start=start, samples=len(original), mean=float(mean), peak=float(peak)))
    arrays = {key: np.concatenate([r[key] for r in rendered]) for key in rendered[0]}
    arrays.update(magnitudes=np.concatenate(magnitudes), durations=np.concatenate(durations))
    for name, value in zip(
        ("modal_gains", "modal_frequencies", "modal_dampings"),
        dsp.modal_controls(model.gains, model.frequencies, model.dampings, model._config["hz_max"]),
    ):
        arrays[name] = np.asarray(value)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "inferred.npz", **arrays)
    wavfile.write(output / "audio.wav", rate, arrays["audio"])
    report = dict(
        checkpoint_sha256=data.digest(checkpoint),
        recording_sha256=data.digest(recording),
        audio_rate=rate,
        force_rate=model._rate,
        control_frames_per_clip=model._config["frames"],
        transforms=transforms,
        output_units="Per-clip normalized audio, uncalibrated force",
        clip_boundary="Independent bounded clips; final clip zero-padded after normalization",
        adaptations=metadata["adaptations"],
    )
    (output / "inference.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()
    infer(args.checkpoint, args.audio, args.output, args.channel)
