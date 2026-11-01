"""Recording selection, preprocessing and provenance for DiffImpact experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_reference(root):
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    for name, expected in protocol["sources"].items():
        if digest(root / name) != expected:
            raise ValueError(f"DiffImpact reference changed: {name}")


def normalize(audio):
    audio = audio.astype(np.float64)
    audio = audio - audio.mean()
    peak = np.max(np.abs(audio))
    if peak == 0 or not np.isfinite(peak):
        raise ValueError("Recording is silent or non-finite")
    return audio / peak


def hammer(root):
    clips, forces, sources = [], [], []
    for name in ("ceramic", "polycarb", "steel", "wood"):
        path = root / "data/hammer" / (name + ".wav")
        rate, audio = wavfile.read(path)
        if rate != 44100 or audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError(f"Unexpected hammer recording: {path}")
        audio = audio[int(0.15 * rate) : int(3.15 * rate)]
        if len(audio) != 3 * rate:
            raise ValueError(f"Truncated hammer recording: {path}")
        clips.append(normalize(audio[:, 0]))
        forces.append(audio[:, 1])
        sources.append(dict(path=str(path), sha256=digest(path), object=name))
    return {"audio": np.array(clips, np.float32), "measured_force": np.array(forces)}, sources


def separation(root):
    tools, objects, recordings = {}, {}, []
    for path in sorted((root / "data/robot").glob("*/*.wav")):
        material, tool, body_material, body = path.parent.name.split("-")
        tool_id = tools.setdefault(f"{material}-{tool}", len(tools))
        object_id = objects.setdefault(f"{body_material}-{body}", len(objects))
        rate, audio = wavfile.read(path)
        if rate != 44100:
            raise ValueError(f"Unexpected robot sample rate: {path}")
        audio = audio[:, 0] if audio.ndim == 2 else audio
        peak = np.max(np.abs(audio.astype(np.float64) - audio.mean()))
        clip = normalize(audio)
        start = int(max(0, np.argmax(np.abs(clip)) - 0.1 * rate))
        recordings.append((clip[start:], tool_id, object_id, peak, path))
    if len(tools) != 4 or len(objects) != 4:
        raise ValueError("Expected all four released tools and objects")
    size = min(44100, min(len(r[0]) for r in recordings))
    selected = []
    for tool in range(4):
        for body in range(4):
            candidates = [r for r in recordings if r[1:3] == (tool, body)]
            if len(candidates) < 2:
                raise ValueError("Missing recordings for a tool/object pair")
            # Match the notebook, including its ordering within each pair.
            order = np.argsort(np.array([r[3] for r in candidates], np.float32))[-2:]
            selected.extend(candidates[i] for i in order)
    sources = [
        dict(path=str(r[4]), sha256=digest(r[4]), tool=list(tools)[r[1]], object=list(objects)[r[2]]) for r in selected
    ]
    return (
        dict(
            audio=np.array([r[0][:size] for r in selected], np.float32),
            tools=np.array([r[1] for r in selected], np.int32),
            objects=np.array([r[2] for r in selected], np.int32),
            indices=np.arange(len(selected), dtype=np.int32),
        ),
        sources,
    )


def asmr(manifest):
    """Load explicit, disjoint train/validation/test intervals for one object."""
    spec = json.loads(manifest.read_text())
    if spec["rate"] != 44100 or not spec["object"]:
        raise ValueError("ASMR manifests require a named object and 44100 Hz audio")
    size = spec["samples"]
    if size <= 0 or size % 441:
        raise ValueError("Clip length must be a positive multiple of 10 ms")
    groups = {"train": [], "validation": [], "test": []}
    intervals, sources = {}, []
    for item in spec["recordings"]:
        path = (manifest.parent / item["path"]).resolve()
        if digest(path) != item["sha256"]:
            raise ValueError(f"ASMR recording hash mismatch: {path}")
        rate, audio = wavfile.read(path)
        start, stop, split = item["start"], item["stop"], item["split"]
        if rate != spec["rate"] or not 0 <= start < stop <= len(audio) or split not in groups:
            raise ValueError(f"Invalid ASMR interval: {item}")
        for a, b in intervals.setdefault(item["sha256"], []):
            if start < b and a < stop:
                raise ValueError("ASMR intervals overlap; train/test leakage is not permitted")
        intervals[item["sha256"]].append((start, stop))
        if audio.ndim == 2:
            audio = audio[:, item["channel"]]
        for offset in range(start, stop - size + 1, size):
            groups[split].append(normalize(audio[offset : offset + size]))
        sources.append(dict(item, path=str(path)))
    if any(not values for values in groups.values()):
        raise ValueError("Each split needs at least one complete clip")
    return {name: {"audio": np.array(values, np.float32)} for name, values in groups.items()}, sources, spec
