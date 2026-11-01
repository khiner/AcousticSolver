"""Convert pinned public CDPAM weights and capture an independent CPU reference once."""

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import torch

WHEEL_SHA256 = "abd860166a55ada94cb653997deed481b518e3a593e9b417a969f99522009eba"
CHECKPOINT_SHA256 = "453c8b6edee1a94f0120236156436ff28fe4d8d884485e4a67695c8e8570bdfe"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/reference/cdpam"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wheel = args.output / "cdpam-0.0.6-py3-none-any.whl"
    if not wheel.exists():
        with urllib.request.urlopen("https://pypi.org/pypi/cdpam/0.0.6/json", timeout=60) as response:
            release = json.load(response)
        url = next(item["url"] for item in release["urls"] if item["digests"]["sha256"] == WHEEL_SHA256)
        urllib.request.urlretrieve(url, wheel)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError("CDPAM wheel hash mismatch")
    with zipfile.ZipFile(wheel) as archive:
        checkpoint = archive.read("cdpam/CDPAM_trained/scratchJNDdefault_best_model.pth")
        source = archive.read("cdpam/models.py")
    if hashlib.sha256(checkpoint).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("CDPAM checkpoint hash mismatch")
    # The pinned 2020 checkpoint uses legacy serialization unsupported by weights_only.
    state = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=False)["state"]
    arrays = {
        k: v.numpy()
        for k, v in state.items()
        if k.startswith(("base_encoder.", "model_dist.")) and not k.endswith("num_batches_tracked")
    }
    np.savez(args.output / "weights.npz", **arrays)
    module_path = args.output / "author_models.py"
    module_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("cdpam_reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    model = reference.FINnet(
        encoder_layers=16,
        encoder_filters=64,
        input_size=512,
        ndim=[16, 6],
        classif_BN=0,
        classif_act="no",
        classif_dp=0.05,
    ).eval()
    model.load_state_dict(state)
    rng = np.random.default_rng(714)
    a = rng.normal(0, 3000, (2, 4097)).astype(np.float32)
    b = (0.8 * a + rng.normal(0, 1000, a.shape)).astype(np.float32)
    with torch.no_grad():

        def embedding(x):
            return torch.nn.functional.normalize(model.base_encoder(torch.from_numpy(x)[:, None])[1], dim=1)

        ea, eb = embedding(a), embedding(b)
        distance = model.model_dist(ea, eb)
    np.savez(
        args.output / "reference.npz",
        a=a,
        b=b,
        embedding_a=ea.numpy(),
        embedding_b=eb.numpy(),
        distance=distance.numpy(),
    )
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(np.asarray(arrays[name], dtype="<f4").tobytes())
    (args.output / "provenance.json").write_text(
        json.dumps(
            dict(
                wheel_sha256=WHEEL_SHA256,
                checkpoint_sha256=CHECKPOINT_SHA256,
                native_weights_sha256=digest.hexdigest(),
                reference="cdpam 0.0.6 author CPU float32 inference",
                preprocessing_adaptation="scipy polyphase resampling instead of librosa/resampy; rounded PCM scale preserved",
            ),
            indent=2,
        )
        + "\n"
    )
    print(args.output / "weights.npz")


if __name__ == "__main__":
    main()
