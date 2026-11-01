"""Stage, launch, inspect, and retrieve the CUDA reference over RunPod SSH."""

import argparse
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/workspace/stable-modal-synthesis"
COMMIT = json.loads(Path(__file__).with_name("protocol.json").read_text())["commit"]


def add_harness(archive, prefix=""):
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.is_file():
            archive.add(path, arcname=prefix + "script/stable_modal_synthesis/" + path.name)
    for path in sorted((ROOT / "src/StableModal").iterdir()):
        if path.is_file():
            archive.add(path, arcname=prefix + "src/StableModal/" + path.name)
    archive.add(ROOT / "src/Metal/FloatPair.h", arcname=prefix + "src/Metal/FloatPair.h")


def main(args):
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-p",
        str(args.port),
    ]
    if args.identity:
        ssh += ["-i", str(args.identity.expanduser()), "-o", "IdentitiesOnly=yes"]
    ssh += [args.host]

    def run(command, **kwargs):
        return subprocess.run([*ssh, command], check=True, **kwargs)

    if args.action == "stage":
        run(
            "python3 -c \"import shutil; assert shutil.disk_usage('/workspace').free > 10*2**30, 'Need at least 10 GiB free for setup'\""
        )
        run(f"mkdir -p {REMOTE}/harness")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            add_harness(tar)
        run(f"tar --no-same-owner --no-same-permissions -xzf - -C {REMOTE}/harness", input=archive.getvalue())
        script = f"""set -eu
cd {REMOTE}
if [ ! -d reference/.git ]; then
  git clone --depth 1 https://github.com/victorzheleznov/jaes-modal-node.git reference
fi
git -C reference fetch --depth 1 origin {COMMIT}
git -C reference checkout {COMMIT}
python3 -m venv --system-site-packages env
env/bin/python -m pip install -r harness/script/stable_modal_synthesis/requirements.txt > install.log 2>&1
env/bin/python -m pip freeze > environment.txt
nvidia-smi -q > gpu.txt
"""
        run("bash -s", input=script.encode())
    elif args.action == "launch":
        if args.epochs is not None and args.epochs < 1:
            raise ValueError("Epoch count must be positive")
        epoch_arg = f" --epochs {args.epochs}" if args.epochs is not None else ""
        free_gib = 180 if args.start_at == "prepare" else 20
        run(
            f"python3 -c \"import shutil; assert shutil.disk_usage('/workspace').free > {free_gib}*2**30, 'Need at least {free_gib} GiB free'\""
        )
        script = f"""set -eu
cd {REMOTE}
if [ -f job.pid ] && grep -aq 'script.stable_modal_synthesis.cuda_job' "/proc/$(cat job.pid)/cmdline" 2>/dev/null; then
  echo 'Job is already running'; exit 1
fi
export PYTHONPATH={REMOTE}/harness
export STABLE_MODAL_REFERENCE={REMOTE}/reference
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
nohup env/bin/python -u -m script.stable_modal_synthesis.cuda_job --phase {args.start_at}{epoch_arg} > job.log 2>&1 < /dev/null &
echo $! > job.pid
cat job.pid
"""
        run("bash -s", input=script.encode())
    elif args.action == "status":
        run(f"cd {REMOTE} && tail -20 job.log && cat status.json")
    elif args.action == "fetch":
        args.output.mkdir(parents=True, exist_ok=True)
        transport = shlex.join(ssh[:-1])
        for name in ("results", "status.json", "job.log", "environment.txt", "gpu.txt"):
            subprocess.run(
                ["rsync", "-az", "-e", transport, f"{args.host}:{REMOTE}/{name}", str(args.output)], check=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "launch", "status", "fetch"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--identity", type=Path)
    parser.add_argument(
        "--start-at",
        choices=("prepare", "train", "evaluate"),
        default="prepare",
        help="Launch at a completed phase boundary after preserving the previous attempt logs",
    )
    parser.add_argument("--epochs", type=int, help="Override the training epoch budget for launch")
    parser.add_argument("--output", type=Path, default=ROOT / "build/stable-modal-cuda")
    main(parser.parse_args())
