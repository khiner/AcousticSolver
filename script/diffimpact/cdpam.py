"""Frozen CDPAM 0.0.6 inference on MLX, using the author's trained weights."""

import hashlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.signal import resample_poly


def weight_digest(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(np.asarray(arrays[name], dtype="<f4").tobytes())
    return digest.hexdigest()


class CDPAM:
    def __init__(self, path):
        from pathlib import Path
        import json

        protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())["cdpam"]
        with np.load(path, allow_pickle=False) as archive:
            if weight_digest(archive) != protocol["native_weights_sha256"]:
                raise ValueError("CDPAM weights do not match the pinned release")
            self.weights = {k: mx.array(archive[k]) for k in archive.files}

    def embedding(self, pcm):
        z = mx.array(pcm)[..., None]
        w = self.weights
        for i in range(16):
            conv, bn = f"base_encoder.encoder.{i}.", f"base_encoder.ebatch.{i}."
            z = mx.conv1d(z, w[conv + "weight"].transpose(0, 2, 1), padding=7) + w[conv + "bias"]
            z = (z - w[bn + "running_mean"]) * mx.rsqrt(w[bn + "running_var"] + 1e-5)
            z = nn.leaky_relu(z * w[bn + "weight"] + w[bn + "bias"], 0.1)
            if (i + 1) % 4 == 0:
                z = z[:, ::2]
        z = mx.mean(z, axis=1)[:, :512]
        return z / mx.maximum(mx.sqrt(mx.sum(z * z, axis=1, keepdims=True)), 1e-12)

    def distance(self, a, b):
        result = mx.zeros((len(a),))
        for i in range(4):
            name = f"model_dist.convs.{i}.0."
            a = a @ self.weights[name + "weight"].T + self.weights[name + "bias"]
            b = b @ self.weights[name + "weight"].T + self.weights[name + "bias"]
            if i < 3:
                a, b = nn.leaky_relu(a, 0.01), nn.leaky_relu(b, 0.01)
            result = result + mx.sum(mx.abs((b - a) * self.weights[f"model_dist.chan_w.{i}"]), axis=1)
        return result

    def __call__(self, a, b):
        # The published loader uses 22050 Hz rounded PCM-scale floats.
        def encode(audio):
            pcm = np.round(resample_poly(audio, 1, 2, axis=-1) * 32768).astype(np.float32)
            return self.embedding(pcm)

        return np.asarray(self.distance(encode(a), encode(b)))
