"""Build the pinned ASMR tapping dataset from archived source videos."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import wavfile

from . import data


def build(spec_path, sources, output):
    spec = json.loads(spec_path.read_text())
    rate, size = spec["rate"], spec["samples"]
    splits = spec["split_seconds"]
    if (
        set(splits) != {"train", "validation", "test"}
        or rate != 44100
        or size <= 0
        or size % 441
        or spec["guard_seconds"] < 0
    ):
        raise ValueError("Expected three splits, 44100 Hz and clips in multiples of 10 ms")
    ordered = sorted(splits.values())
    if any(a < 0 or b <= a or (b - a) * rate % size for a, b in ordered):
        raise ValueError("Each split must contain a positive whole number of clips")
    if any(a[1] + spec["guard_seconds"] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Split intervals overlap or omit the required guard gap")
    duration = ordered[-1][1]
    archives, used = {}, {}
    for source in spec["sources"]:
        path = sources / (source["id"] + ".mp4")
        if data.digest(path) != source["sha256"]:
            raise ValueError(f"Archived video hash mismatch: {path}")
        info = json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], text=True
            )
        )
        audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
        if int(audio["sample_rate"]) != rate or audio["channels"] != 2:
            raise ValueError("Pinned sources require stereo 44100 Hz audio")
        archives[source["id"]] = (path, source, float(info["format"]["duration"]))
    names = set()
    for item in spec["objects"]:
        name, start, key = item["name"], item["start_seconds"], item["source"]
        if name in names or Path(name).name != name or name in (".", ".."):
            raise ValueError("Object names must be unique filenames")
        names.add(name)
        if not 0 <= start < start + duration <= archives[key][2]:
            raise ValueError("Object interval is outside its source video")
        if any(start < end and begin < start + duration for begin, end in used.setdefault(key, [])):
            raise ValueError("Different objects cannot reuse the same source interval")
        used[key].append((start, start + duration))
    output.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    summary = dict(
        dataset=spec["name"],
        specification_sha256=data.digest(spec_path),
        ffmpeg=version,
        sources=spec["sources"],
        objects={},
        complete_paper_reproduction=False,
        adaptations=spec["adaptations"],
    )
    for item in spec["objects"]:
        path, source, _ = archives[item["source"]]
        name, start = item["name"], item["start_seconds"]
        wav = output / (name + ".wav")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-af",
                f"atrim=start_sample={start * rate}:end_sample={(start + duration) * rate},asetpts=PTS-STARTPTS",
                "-c:a",
                "pcm_f32le",
                str(wav),
            ],
            check=True,
        )
        actual_rate, audio = wavfile.read(wav)
        if actual_rate != rate or audio.shape != (duration * rate, 2) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid decoded recording: {wav}")
        checksum = data.digest(wav)
        manifest = dict(
            object=name,
            rate=rate,
            samples=size,
            dataset=spec["name"],
            adaptations=spec["adaptations"],
            role=item["role"],
            source=dict(
                source,
                start_seconds=start,
                stop_seconds=start + duration,
                ffmpeg=version,
                audio_video_alignment="WAV sample zero corresponds to start_seconds in archived video",
            ),
            recordings=[
                dict(path=wav.name, sha256=checksum, split=split, channel=0, start=a * rate, stop=b * rate)
                for split, (a, b) in splits.items()
            ],
        )
        manifest_path = output / (name + ".json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        groups, _, _ = data.asmr(manifest_path)
        summary["objects"][name] = dict(
            manifest=manifest_path.name,
            sha256=data.digest(manifest_path),
            wav_sha256=checksum,
            clips={k: len(v["audio"]) for k, v in groups.items()},
            peak=float(np.max(np.abs(audio))),
            rms=float(np.sqrt(np.mean(audio**2))),
            channel_for_training=0,
            role=item["role"],
        )
    (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path(__file__).resolve().parents[2] / "config/diffimpact/asmr-v1.json"
    )
    parser.add_argument("--sources", type=Path, default=Path("data/diffimpact/sources"))
    parser.add_argument("--output", type=Path, default=Path("data/diffimpact/asmr-v1"))
    args = parser.parse_args()
    result = build(args.spec, args.sources, args.output)
    print(json.dumps({name: item["clips"] for name, item in result["objects"].items()}))
