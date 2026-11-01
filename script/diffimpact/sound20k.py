"""Index isolated SoundSynth object tracks and physical labels for Appendix D."""

import argparse
import configparser
import json
from pathlib import Path

from .data import digest


def index(audio_root, reference):
    labels, sources = {}, {}
    entry = reference / "data/data_entry.txt"
    sources[str(entry.relative_to(reference))] = digest(entry)
    allowed = set()
    for line in entry.read_text().splitlines():
        row = list(map(int, line.split()))
        allowed.add((f"scene-{row[0]}", "obj-" + "-".join(map(str, row[1::2])), "mat-" + "-".join(map(str, row[2::2]))))
    records = []
    for path in sorted(audio_root.rglob("*.wav")):
        if not path.stem.isdigit():
            continue
        parents = path.parts
        candidates = [tuple(parents[i : i + 3]) for i, part in enumerate(parents) if part.startswith("scene-")]
        if len(candidates) != 1 or candidates[0] not in allowed:
            raise ValueError(f"Track has no matching SoundSynth metadata entry: {path}")
        scene, objects, materials = candidates[0]
        object_ids = list(map(int, objects.split("-")[1:]))
        material_ids = list(map(int, materials.split("-")[1:]))
        position = int(path.stem)
        if position >= len(material_ids):
            raise ValueError(f"Track index exceeds scene object count: {path}")
        if object_ids[position] == 101:
            continue
        material = material_ids[position]
        if material not in labels:
            material_path = reference / f"config/material/material-{material}.cfg"
            config = configparser.ConfigParser()
            config.read(material_path)
            values = config["DEFAULT"]
            if not {"name", "youngs"} <= values.keys():
                raise ValueError(f"Missing material labels: {material_path}")
            labels[material] = dict(material=values["name"], youngs_pa=float(values["youngs"]))
            sources[str(material_path.relative_to(reference))] = digest(material_path)
        records.append(
            dict(
                path=str(path.resolve()),
                sha256=digest(path),
                group="/".join((scene, objects, materials, str(position))),
                **labels[material],
            )
        )
    if not records:
        raise ValueError("No isolated Sound-20K tracks found; merged mixtures cannot be assigned an object label")
    return dict(
        dataset="Sound-20K",
        recordings=records,
        provenance=dict(
            repository="https://github.com/ztzhang/SoundSynth",
            sources=sources,
            indexing="SoundSynth scene/obj/mat paths and zero-based isolated track index; exclude floor 101",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("external/soundsynth"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = index(args.audio, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + "\n")
