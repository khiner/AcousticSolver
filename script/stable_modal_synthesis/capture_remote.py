"""Capture portable gradient fixtures on any SSH-accessible CUDA machine."""

import argparse
import io
import json
import shlex
import subprocess
import tarfile
import uuid
from pathlib import Path

from . import reference as ref
from .training_reference import load_fixture
from .runpod import add_harness


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(args.port),
        args.host,
    ]
    root = args.remote_root.rstrip("/") + "/" + uuid.uuid4().hex[:12]
    quote = shlex.quote

    def remote(command, **kwargs):
        return subprocess.run([*ssh, command], check=True, **kwargs)

    archive = io.BytesIO()
    inputs = []
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name, digest in ref.PROTOCOL["sources"].items():
            path = args.reference / name
            if ref.digest(path) != digest:
                raise ValueError(f"Pinned source changed: {name}")
            tar.add(path, arcname="reference/" + name)
        add_harness(tar, "harness/")
        for i, path in enumerate(args.inputs):
            fixture = load_fixture(path)
            name = f"case-{i}.pt.gz" if path.name.endswith(".gz") else f"case-{i}.pt"
            inputs.append(dict(name=name, source=str(path), sha256=ref.digest(path), steps=len(fixture["steps"])))
            tar.add(path, arcname="inputs/" + name)
    ref.write_json(args.output / "request.json", dict(host=args.host, port=args.port, remote_root=root, inputs=inputs))
    print(f"Staging {len(archive.getbuffer())/1e6:.1f} MB; no original pod or dataset storage required", flush=True)
    remote(
        f"mkdir -p {quote(root)} && tar --no-same-owner --no-same-permissions -xzf - -C {quote(root)}",
        input=archive.getvalue(),
    )
    print("Installing pinned dependencies over the existing CUDA PyTorch environment", flush=True)
    setup = f"""set -eu
cd {quote(root)}
python3 -m venv --system-site-packages env
env/bin/python -m pip install -r harness/script/stable_modal_synthesis/requirements.txt > install.log 2>&1
env/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
mkdir output
env/bin/python -m pip freeze > output/environment.txt
nvidia-smi -q > output/gpu.txt
"""
    remote("bash -s", input=setup.encode())
    for i, row in enumerate(inputs):
        print(f'Capturing case {i}: {row["steps"]} updates', flush=True)
        command = (
            f"cd {quote(root)} && PYTHONPATH=harness env/bin/python -u -m script.stable_modal_synthesis.training_reference "
            f'--reference reference --inputs {quote("inputs/"+row["name"])} --output output/case-{i} '
            f'--device cuda --steps {row["steps"]}'
        )
        remote(command)
    manifest = """from pathlib import Path
import hashlib,json
r=Path('output')
d={str(p.relative_to(r)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in r.rglob('*') if p.is_file()}
(r/'manifest.json').write_text(json.dumps(d,indent=2)+'\\n')
"""
    remote(f"cd {quote(root)} && env/bin/python -c {quote(manifest)}")
    print("Downloading and verifying all captures", flush=True)
    download = args.output / "download.tar.gz"
    with download.open("wb") as stream:
        remote(f'tar -czf - -C {quote(root+"/output")} .', stdout=stream)
    with tarfile.open(download) as tar:
        tar.extractall(args.output, filter="data")
    manifest = json.loads((args.output / "manifest.json").read_text())
    for name, expected in manifest.items():
        if ref.digest(args.output / name) != expected:
            raise RuntimeError(f"Download checksum mismatch: {name}")
    download.unlink()
    print(
        f"Verified {len(manifest)} captured artifacts locally; remote machine is no longer needed for these tests",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--reference", type=Path, default=ref.DEFAULT_REFERENCE)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root", default="/workspace/stable-modal-gradient-capture")
    run(parser.parse_args())
