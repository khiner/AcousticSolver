"""Run Python tools in their pinned local environments."""

import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def ensure_environment(name, requirements):
    directory = ROOT / "build/reference" / name
    python = directory / "bin/python"
    if Path(sys.prefix).resolve() != directory:
        if not python.exists():
            executable = shutil.which("python3.12") or sys.executable
            subprocess.run([executable, "-m", "venv", str(directory)], check=True)
        os.execv(str(python), [str(python), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])
    requirements = ROOT / requirements
    packages = [line.split("==") for line in requirements.read_text().splitlines() if line and not line.startswith("#")]
    try:
        ready = all(importlib.metadata.version(package) == version for package, version in packages)
    except importlib.metadata.PackageNotFoundError:
        ready = False
    if not ready:
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)


def run_module(name, requirements, module):
    import runpy

    ensure_environment(name, requirements)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    runpy.run_module(module, run_name="__main__")
