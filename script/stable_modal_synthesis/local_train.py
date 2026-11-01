"""Run bounded local training with author batching and resumable Accelerate updates."""

import argparse
import json
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from torch.utils.data.distributed import DistributedSampler

from . import reference as ref
from .local_data import DEFAULT_DATA, Recordings, save
from .native import SOURCE, SCRIPT, memory
from .training import Trainer, parameters, training_inputs
from .training_reference import load_fixture

DEFAULT_FIXTURE = ref.ROOT / "tests/fixtures/stable_modal_synthesis/initial.pt.gz"


class Run:
    def __init__(self, data, fixture, output, reference=ref.DEFAULT_REFERENCE, chunk=256, resume=None):
        if not 1 <= chunk <= 2048:
            raise ValueError("Chunk must contain 1..2048 windows")
        self.cfg = ref.load(reference)
        self.output, self.chunk = Path(output), chunk
        manifest_path = Path(data) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.datasets = {split: Recordings(data, manifest, split) for split in ("train", "valid")}
        self.data = dict(
            manifest=str(manifest_path.resolve()),
            generation=manifest["generation"],
            scope=manifest["scope"],
            split_counts={split: len(dataset) for split, dataset in self.datasets.items()},
        )
        if len(self.datasets["train"]) % 4:
            raise ValueError("Training recordings must form complete four-recording batches")
        self.collate = instantiate(self.cfg.collate_func)
        self.sampler = DistributedSampler(self.datasets["train"], num_replicas=1, rank=0, shuffle=True, seed=0)
        self.model = ref.learned(self.cfg)
        initial = load_fixture(fixture)
        self.settings = initial["adam_settings"]
        self.settings["betas"] = tuple(self.settings["betas"])
        if self.settings != dict(lr=0.001, betas=(0.9, 0.999), epsilon=1e-8):
            raise ValueError("Expected the released Adam settings")
        self.contract = dict(
            dataset_sha256=ref.digest(manifest_path),
            fixture_sha256=ref.digest(fixture),
            protocol_sha256=ref.digest(SCRIPT / "protocol.json"),
            chunk=chunk,
            recordings_per_update=4,
            sampler="DistributedSampler seed 0; set_epoch starting at 1; consecutive groups of four",
            sources={
                path.name: ref.digest(path)
                for path in [
                    *(
                        SCRIPT / name
                        for name in (
                            "local_train.py",
                            "local_data.py",
                            "training.py",
                            "training_reference.py",
                            "native.py",
                            "reference.py",
                        )
                    ),
                    SOURCE / "TrainAccelerate.cpp",
                ]
            },
        )
        checkpoint = dict(
            model_state_dict=initial["model_state"], adam=initial["adam"], batch=0, epoch_loss=0.0, history=[]
        )
        if resume:
            if Path(resume).resolve().parent != self.output.resolve():
                raise ValueError("Resume in the existing run directory to retain its best checkpoint")
            checkpoint = torch.load(resume, weights_only=True, map_location="cpu")
            if checkpoint["contract"] != self.contract:
                raise ValueError("Resume requires identical data, initialization, code and accumulation settings")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.adam = ref.cpu(checkpoint["adam"])
        self.batch, self.epoch_loss, self.history = checkpoint["batch"], checkpoint["epoch_loss"], checkpoint["history"]
        self.output.mkdir(parents=True, exist_ok=bool(resume))
        self.trainer = Trainer()

    def checkpoint(self):
        return dict(
            contract=self.contract,
            model_state_dict=ref.cpu(self.model.state_dict()),
            adam=ref.cpu(self.adam),
            batch=self.batch,
            epoch_loss=self.epoch_loss,
            history=self.history,
        )

    @property
    def epoch(self):
        return len(self.history) + 1

    def batches(self, split):
        if split == "train":
            self.sampler.set_epoch(self.epoch)
            order = list(self.sampler)
        else:
            order = list(range(len(self.datasets[split])))
        return [order[start : start + 4] for start in range(0, len(order), 4)]

    def evaluate(self, split, indices):
        started = time.perf_counter()
        instances = [self.datasets[split][i] for i in indices]
        loaded = time.perf_counter()
        batch = ref.double_batch(self.collate(instances))
        del instances
        prepared = training_inputs(self.model, batch)
        ready = time.perf_counter()
        result = self.trainer.accumulate(parameters(self.model), prepared, self.chunk, backward=(split == "train"))
        result["timing"] = dict(
            load_seconds=loaded - started, prepare_seconds=ready - loaded, compute_seconds=time.perf_counter() - ready
        )
        return result, len(prepared["initial"])

    def validate(self):
        # The author selects checkpoints using teacher-forced validation loss, not full-trajectory audio error.
        total, timings = 0.0, []
        for indices in self.batches("valid"):
            result, _ = self.evaluate("valid", indices)
            total += result["loss"] * len(indices)
            timings.append(result["timing"])
        return total / len(self.datasets["valid"]), {key: sum(t[key] for t in timings) for key in timings[0]}

    def step(self):
        started = time.perf_counter()
        groups = self.batches("train")
        indices = groups[self.batch]
        result, windows = self.evaluate("train", indices)
        adam_started = time.perf_counter()
        self.trainer.adam(parameters(self.model), result["gradient"], self.adam, **self.settings)
        row = dict(
            epoch=self.epoch,
            batch=self.batch,
            optimizer_step=self.adam["step"],
            recordings=[self.datasets["train"].records[i]["index"] for i in indices],
            windows=windows,
            loss=result["loss"],
            update_seconds=time.perf_counter() - started,
            timing=dict(result["timing"], adam_seconds=time.perf_counter() - adam_started),
        )
        self.epoch_loss += result["loss"]
        self.batch += 1
        improved = False
        if self.batch == len(groups):
            validation_started = time.perf_counter()
            valid, row["validation_timing"] = self.validate()
            row["validation_seconds"] = time.perf_counter() - validation_started
            improved = not self.history or valid < min(entry["valid_loss"] for entry in self.history)
            self.history.append(dict(epoch=self.epoch, train_loss=self.epoch_loss / len(groups), valid_loss=valid))
            self.batch, self.epoch_loss = 0, 0.0
            row["validation_loss"] = valid
        checkpoint_started = time.perf_counter()
        checkpoint = self.checkpoint()
        save(self.output / "last.pt", checkpoint)
        if improved:
            save(self.output / "best.pt", checkpoint)
        row["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
        row["total_seconds"] = time.perf_counter() - started
        ref.write_json(
            self.output / "progress.json", dict(last_update=row, history=self.history, contract=self.contract)
        )
        print(row, flush=True)
        return row, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--reference", type=Path, default=ref.DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--updates", type=int, required=True, help="Explicit bounded update budget; does not restart the CUDA run"
    )
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    started = time.perf_counter()
    run = Run(args.data, args.fixture, args.output, args.reference, args.chunk, args.resume)
    ready = time.perf_counter()
    rows = []
    for _ in range(args.updates):
        row, _ = run.step()
        rows.append(row)
    ref.write_json(
        args.output / "report.json",
        dict(
            dataset=run.data,
            contract=run.contract,
            updates=rows,
            history=run.history,
            startup_seconds=ready - started,
            loop_seconds=time.perf_counter() - ready,
            memory=memory(),
            scope="This bounded invocation; loop includes loading, preparation, training, validation and checkpoint writes",
        ),
    )
