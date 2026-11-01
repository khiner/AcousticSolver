"""Check the pinned notebook preprocessing, spectral loss and contact batching."""

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np

from script.diffimpact import run as diffimpact
from .data import write_json

ROOT = Path(__file__).resolve().parents[2]


def check(checkout, output):
    ns = diffimpact.reference(checkout)
    tf, ddsp = ns["tf"], ns["ddsp"]
    notebook = json.loads((checkout / "AnalysisBySynthesis.ipynb").read_text())
    data = ast.parse("".join(notebook["cells"][2]["source"]))
    data.body[0].value = ast.Constant(str(checkout / "data/hammer"))
    data.body.pop()
    import scipy.io

    original = dict(os=os, scipy=scipy, np=np)
    exec(compile(ast.fix_missing_locations(data), "author-data", "exec"), original)
    np.testing.assert_array_equal(original["clips"], ns["clips"])
    np.testing.assert_array_equal(original["forces"], ns["forces"])
    assert len(ns["variables"]) == 8 and ns["frequencies"].shape == (4, 1, 256)
    assert all(v is not ns["reverb_gains"] for v in ns["variables"])
    # Independent DFT construction checks the notebook's linear-plus-log L1 convention.
    target = np.random.default_rng(42).normal(0, 0.2, (1, 2200)).astype(np.float32)
    prediction = target * 0.7
    expected = 0.0
    for size in (2048, 1024, 512, 256):
        hop = size // 4
        starts = range(0, target.shape[1], hop)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
        spectra = []
        for audio in (target[0], prediction[0]):
            padded = np.pad(audio, (0, size))
            spectra.append(abs(np.fft.rfft(np.stack([padded[s : s + size] for s in starts]) * window)))
        a, b = spectra
        expected += np.mean(abs(a - b)) + np.mean(abs(np.log(np.maximum(a, 1e-5)) - np.log(np.maximum(b, 1e-5))))
    actual = float(ns["loss"].call(target, prediction))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    impact = ddsp.synths.Impact(n_samples=2000, sample_rate=20000, max_impact_frequency=20, include_noise=False)
    magnitudes = tf.constant([[[1.0], [0.2]], [[0.3], [2.0]]])
    taus = tf.constant([[[0.001], [0.001]], [[0.002], [0.002]]])
    batched = impact.get_signal(magnitudes, taus).numpy()[1]
    independent = impact.get_signal(magnitudes[1:], taus[1:]).numpy()[0]
    report = dict(
        passed=True,
        exact_author_preprocessing=True,
        independent_spectral_loss=actual,
        upstream_contact_batch_relative_difference=float(
            np.linalg.norm(batched - independent) / np.linalg.norm(independent)
        ),
    )
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "external/diffimpact")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.reference.resolve(), args.output)
