"""Archive the pinned author source and input assets in durable local storage."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

from . import UPSTREAM
from .runtime import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("external/diffsound"))
    parser.add_argument("--output", type=Path, default=Path("data/diffsound/source"))
    args = parser.parse_args()
    manifest = args.output / "manifest.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if record["commit"] != UPSTREAM:
            raise ValueError("Wrong archived upstream revision")
        for relative, expected in record["files"].items():
            if digest(args.output / relative) != expected:
                raise ValueError(f"Archive changed: {relative}")
        print(f'Verified {len(record["files"])} archived files')
        return
    if not (args.source / ".git").exists():
        raise FileNotFoundError("Initialize the pinned reference: git submodule update --init external/diffsound")
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM:
        raise ValueError(f"Expected upstream {UPSTREAM}, found {revision}")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.source / "data", args.output / "data")
    with ZipFile(args.output / "implementation.zip", "w", ZIP_DEFLATED) as archive:
        for folder in ("src", "experiments", "configs"):
            for path in sorted((args.source / folder).rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(args.source))
    files = {str(p.relative_to(args.output)): digest(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    manifest.write_text(json.dumps(dict(commit=UPSTREAM, files=files), indent=2) + "\n")
    print(f"Archived {len(files)} files to {args.output}")


if __name__ == "__main__":
    main()
