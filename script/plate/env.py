"""Isolated, pinned dependencies for the unchanged FA2026 reference."""
import os
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "external/fa2026"
REVISION = "7b1e0a2088c9f5758a216c218f9c34e87656d100"
DEPENDENCIES = ["numpy==2.4.4", "scipy==1.17.1", "numba==0.65.0"]


def ensure_environment(script):
    directory = ROOT / "build/plate-reference"
    python = directory / "bin/python"
    if Path(sys.prefix).resolve() == directory.resolve():
        try:
            ready = all(importlib.metadata.version(name) == version for name, version in (p.split("==") for p in DEPENDENCIES))
        except importlib.metadata.PackageNotFoundError:
            ready = False
        if not ready:
            subprocess.run([str(python), "-m", "pip", "install", *DEPENDENCIES], check=True)
        return
    if not python.exists():
        executable = shutil.which("python3.12") or sys.executable
        subprocess.run([executable, "-m", "venv", str(directory)], check=True)
    os.execv(str(python), [str(python), str(script), *sys.argv[1:]])


def require_reference():
    if not (REFERENCE / "src/generators/plate.py").is_file():
        raise RuntimeError("Initialize reference: git submodule update --init external/fa2026")
    commit = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip()
    if commit != REVISION:
        raise RuntimeError(f"Reference revision is {commit}, expected {REVISION}")
    dirty = subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError("Reference tracked files are modified; restore the pinned checkout before validation")
    # JIT caches remain outside the reference checkout.
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "build/plate-reference/numba-cache")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REFERENCE))
    return commit
