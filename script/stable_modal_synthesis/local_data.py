"""Regenerate and verify local trajectories from archived CUDA-drawn parameters."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, default_collate

from . import reference as ref

DEFAULT_DATA = ref.ROOT / "data/stable_modal_synthesis/pilot"
DEFAULT_METADATA = ref.ROOT / "tests/fixtures/stable_modal_synthesis/pilot-metadata.json"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def prepare(args):
    if args.batch_size < 1:
        raise ValueError("Generation batch size must be positive")
    cfg = ref.load(args.reference)
    from src.models import ModalSystem

    metadata = json.loads(args.metadata.read_text())
    if metadata["author_commit"] != ref.PROTOCOL["commit"]:
        raise ValueError("Metadata and author source revisions differ")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        complete=False,
        metadata=metadata,
        metadata_sha256=ref.digest(args.metadata),
        provenance=ref.provenance(args.reference),
        splits={},
        generation="Unchanged float64 CPU author ModalSystem/SAV with archived CUDA modal parameters; no new RNG draws",
        scope="CPU regeneration adaptation from archived CUDA parameters, not byte-identical CUDA targets",
        split_counts={split: len(rows) for split, rows in metadata["splits"].items()},
        generation_batch_size=args.batch_size,
        audio_relative_tolerance=1e-6,
    )
    for split, rows in metadata["splits"].items():
        records = manifest["splits"][split] = []
        generator = ref.generator(cfg.datasets[split])
        start = time.perf_counter()
        for offset in range(0, len(rows), args.batch_size):
            selected = rows[offset : offset + args.batch_size]
            instances = [
                {k: torch.tensor(v, dtype=torch.float64) if isinstance(v, list) else v for k, v in row.items()}
                for row in selected
            ]
            batch = default_collate(instances)
            print(dict(generating=split, first=offset, recordings=len(selected)), flush=True)
            with torch.no_grad():
                target = ModalSystem.forward(generator, **batch)
            for i, row in enumerate(selected):
                audio = target["w"][i].numpy()
                path = args.reference / "audio" / ref.PROTOCOL["audio_directories"][split] / f'{row["idx"]}.wav'
                published, rate = sf.read(path)
                if rate != row["fs"] or published.shape != audio.shape:
                    raise ValueError("Published audio dimensions differ")
                error = float(np.linalg.norm(audio / np.max(np.abs(audio)) - published) / np.linalg.norm(published))
                if not np.isfinite(error) or error >= manifest["audio_relative_tolerance"]:
                    raise RuntimeError(f'{split}/{row["idx"]} audio comparison failed: {error}')
                name = f'{split}/{row["idx"]}.pt'
                save(
                    args.output / name,
                    dict(instances[i], target_output=target["output"][i].clone(), target_w=target["w"][i].clone()),
                )
                records.append(
                    dict(
                        file=name,
                        index=row["idx"],
                        sha256=ref.digest(args.output / name),
                        published_sha256=ref.digest(path),
                        published_audio_relative_error=error,
                    )
                )
            del target, batch
            ref.write_json(args.output / "manifest.json", manifest)
            print(
                dict(
                    generated=split,
                    completed=len(records),
                    seconds=time.perf_counter() - start,
                    max_audio_relative_error=max(r["published_audio_relative_error"] for r in records),
                ),
                flush=True,
            )
    manifest["complete"] = True
    ref.write_json(args.output / "manifest.json", manifest)


class Recordings(Dataset):
    def __init__(self, directory, manifest, split):
        self.directory = Path(directory)
        self.records = manifest["splits"][split]
        if not manifest["complete"] or not self.records:
            raise ValueError("Expected a complete, nonempty local dataset")
        if len({r["index"] for r in self.records}) != len(self.records):
            raise ValueError("Recording indices must be distinct within each split")
        for record in self.records:
            if ref.digest(self.directory / record["file"]) != record["sha256"]:
                raise ValueError(f'Dataset checksum mismatch: {record["file"]}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return torch.load(self.directory / self.records[index]["file"], weights_only=True, map_location="cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ref.DEFAULT_REFERENCE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Bound generation memory without changing recording parameters"
    )
    prepare(parser.parse_args())
