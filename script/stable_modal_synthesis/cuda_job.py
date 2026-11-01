"""Generate CUDA datasets, reproduce checkpoint metrics, train, and evaluate."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from . import reference as ref


def evaluate(reference, cfg, datasets, checkpoint, directory):
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=ref.digest(checkpoint), splits={})
    model = ref.learned(cfg, checkpoint).cuda().eval()
    for split, dataset in datasets.items():
        rows = []
        per_mode = []
        linear_per_mode = []
        for batch in DataLoader(dataset, batch_size=60, shuffle=False):
            indices = batch["idx"].tolist()
            cpu = batch
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.inference_mode():
                prediction = model(**batch)
                frames = int(batch["fs"][0]) // 10
                for i, index in enumerate(indices):
                    row = dict(index=index)
                    for name, target, output in (
                        ("q", batch["target_output"][i, :75], prediction["output"][i, :75]),
                        ("w", batch["target_w"][i], prediction["w"][i]),
                    ):
                        for suffix, length in (("initial", frames), ("full", target.shape[-1])):
                            row[name + "_" + suffix] = float(
                                (output[..., :length] - target[..., :length]).square().sum()
                                / target[..., :length].square().sum()
                            )
                    rows.append(row)
                    published, _ = sf.read(
                        reference / "audio" / ref.PROTOCOL["audio_directories"][split] / f"{index}_pred.wav"
                    )
                    audio = prediction["w"][i].cpu().numpy()
                    row["published_prediction_relative_l2"] = float(
                        np.linalg.norm(audio / np.max(np.abs(audio)) - published) / np.linalg.norm(published)
                    )
                    if split == "test":
                        linear = dataset._getlin(index)
                        target_w = cpu["target_w"][i, :frames]
                        row["linear_w_initial"] = float(
                            (linear["linear_w"][:frames] - target_w).square().sum() / target_w.square().sum()
                        )
                        target_q = cpu["target_output"][i, :75, :frames]
                        per_mode.append(
                            (prediction["output"][i, :75, :frames].cpu() - target_q).square().mean(-1).numpy()
                        )
                        linear_per_mode.append(
                            (linear["linear_output"][:75, :frames] - target_q).square().mean(-1).numpy()
                        )
                    # Retain full states for a cross-backend case from every partition.
                    if index == 0:
                        torch.save(
                            dict(
                                metadata={k: v for k, v in dataset._meta[index].items() if not k.startswith("target_")},
                                target_output=cpu["target_output"][i].clone(),
                                target_w=cpu["target_w"][i].clone(),
                                output=prediction["output"][i].cpu(),
                                w=prediction["w"][i].cpu(),
                            ),
                            directory / (split + "-case-0.pt"),
                        )
                    sf.write(
                        directory / f"{split}-{index}-pred.wav",
                        prediction["w"][i].cpu().numpy(),
                        int(cpu["fs"][i]),
                        subtype="DOUBLE",
                    )
                if split == "test":
                    selected = dict(
                        worst=max(rows, key=lambda r: r["w_full"])["index"],
                        strongest=max(rows, key=lambda r: r["linear_w_initial"])["index"],
                    )
                    for label, index in selected.items():
                        if index in indices:
                            i = indices.index(index)
                            torch.save(
                                dict(output=prediction["output"][i].cpu(), w=prediction["w"][i].cpu()),
                                directory / (label + ".pt"),
                            )
                    ref.write_json(directory / "figure-selection.json", selected)
            print(json.dumps(dict(evaluating=str(checkpoint), split=split, completed=len(rows))), flush=True)
            ref.write_json(directory / (split + "-per-recording.json"), rows)
        metrics = {name: float(np.mean([row[name] for row in rows])) for name in ref.PROTOCOL["table_2"][split]}
        report["splits"][split] = dict(
            count=len(rows),
            metrics=metrics,
            paper=ref.PROTOCOL["table_2"][split],
            ratio_to_paper={k: metrics[k] / ref.PROTOCOL["table_2"][split][k] for k in metrics},
        )
        ref.write_json(directory / "report.json", report)
        if split == "test":
            np.savez(
                directory / "mode-mse.npz", predicted=np.mean(per_mode, axis=0), linear=np.mean(linear_per_mode, axis=0)
            )
            from .figures import generate

            generate(reference, directory, dataset)
    del model
    torch.cuda.empty_cache()
    return report


def main(args):
    reference = Path(os.environ["STABLE_MODAL_REFERENCE"]).resolve()
    root = reference.parent
    os.chdir(reference)
    results = root / "results"
    results.mkdir(exist_ok=True)
    cfg = ref.load(reference)
    training = reference / "out/reproduction"
    saved_config = training / "cfg.yaml"
    epochs = args.epochs
    if epochs is None:
        epochs = (
            OmegaConf.load(saved_config).num_epochs
            if args.phase != "prepare" and saved_config.exists()
            else cfg.num_epochs
        )
    if epochs < 1:
        raise ValueError("Epoch count must be positive")
    from .cuda_graph import install

    install()
    started = args.started or time.time()
    provenance = ref.provenance(reference)
    provenance.update(
        device=torch.cuda.get_device_name(),
        dataset_rng="cuda",
        training_epochs=epochs,
        author_training_epochs=cfg.num_epochs,
        adaptations=[
            "Python 3.12 instead of author Python 3.11.9",
            "One GPU with batch size four instead of four GPUs with batch size one",
            "Per-epoch durable checkpoints and disabled WandB logging",
            "CUDA Graph capture of unchanged SAV steps for long inference; not used for training gradients",
            "CUDA dataset generation with full-split batches; same parameter draw sequence",
            "Separate process lifetimes for preparation, training, and final evaluation",
        ],
    )
    if epochs != cfg.num_epochs:
        provenance["adaptations"].append(
            f"User-requested training budget of {epochs} epochs instead of {cfg.num_epochs}"
        )
    ref.write_json(
        results / ("provenance.json" if args.phase == "prepare" else args.phase + "-provenance.json"), provenance
    )

    def status(phase, **details):
        ref.write_json(root / "status.json", dict(phase=phase, started=started, updated=time.time(), **details))

    def next_phase(phase):
        # Replace the process to release live tensors, graph pools, and the CUDA context.
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-u",
                "-m",
                "script.stable_modal_synthesis.cuda_job",
                "--phase",
                phase,
                "--started",
                str(started),
                "--epochs",
                str(epochs),
            ],
        )

    try:
        if args.phase == "prepare":
            datasets = {}
            fingerprints = {}
            for split, config in cfg.datasets.items():
                status("generate", split=split)
                datasets[split] = instantiate(config, device="cuda")
                dataset = datasets[split]
                metadata = [
                    {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in row.items()}
                    for row in dataset._meta
                ]
                ref.write_json(results / (split + "-metadata.json"), metadata)
                rows = []
                for i in range(len(dataset)):
                    path = reference / "audio" / ref.PROTOCOL["audio_directories"][split] / f"{i}.wav"
                    published, sr = sf.read(path)
                    generated, rate = sf.read(dataset._audio_dir / f"{i}.wav")
                    if sr != rate or published.shape != generated.shape:
                        raise ValueError("Published/generated audio shape mismatch")
                    published_linear, _ = sf.read(path.with_name(f"{i}_lin.wav"))
                    generated_linear, _ = sf.read(dataset._audio_dir / f"{i}_lin.wav")
                    rows.append(
                        dict(
                            index=i,
                            relative_l2=float(np.linalg.norm(generated - published) / np.linalg.norm(published)),
                            linear_relative_l2=float(
                                np.linalg.norm(generated_linear - published_linear) / np.linalg.norm(published_linear)
                            ),
                            published_sha256=ref.digest(path),
                            generated_sha256=ref.digest(dataset._data_dir / f"{i}.pt"),
                            generated_linear_sha256=ref.digest(dataset._data_dir / f"{i}_lin.pt"),
                        )
                    )
                fingerprints[split] = rows
                ref.write_json(results / "dataset-comparison.json", fingerprints)
                print(
                    json.dumps(
                        dict(dataset=split, published_audio_max_relative_l2=max(r["relative_l2"] for r in rows))
                    ),
                    flush=True,
                )
                if max(max(r["relative_l2"], r["linear_relative_l2"]) for r in rows) > 1e-6:
                    raise RuntimeError(f"{split} regenerated audio differs from release; investigate before training")
                torch.cuda.empty_cache()

            status("evaluate-author-checkpoint")
            evaluate(reference, cfg, datasets, reference / ref.PROTOCOL["checkpoint"], results / "author-checkpoint")
            next_phase("train")

        if args.phase == "train":
            report = json.loads((results / "author-checkpoint/report.json").read_text())
            fingerprints = json.loads((results / "dataset-comparison.json").read_text())
            if report["checkpoint_sha256"] != ref.digest(reference / ref.PROTOCOL["checkpoint"]):
                raise RuntimeError("Released-checkpoint evaluation does not match the pinned checkpoint")
            for split, config in cfg.datasets.items():
                rows = fingerprints[split]
                if len(rows) != config.num_variations or report["splits"][split]["count"] != config.num_variations:
                    raise RuntimeError(f"Incomplete {split} preparation")
                if max(max(r["relative_l2"], r["linear_relative_l2"]) for r in rows) > 1e-6:
                    raise RuntimeError(f"{split} preparation did not pass audio comparison")
            status("train", epochs=epochs)
            training.mkdir(exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=1",
                str(Path(__file__).with_name("cuda_train.py")),
                "--config-path=" + str((reference / ref.PROTOCOL["config"]).parent),
                "--config-name=cfg",
                "batch_size=4",
                "writer.mode=disabled",
                "save_dir=out/reproduction",
                f"num_epochs={epochs}",
            ]
            for split in cfg.datasets:
                command.append(f"+datasets.{split}.device=cuda")
            ref.write_json(results / "training-command.json", command)
            subprocess.run(command, cwd=reference, check=True)
            for name in ("checkpoint_best.pt", f"checkpoint_{epochs}.pt", "cfg.yaml", "history.jsonl", "progress.json"):
                shutil.copy2(training / name, results / name)
            next_phase("evaluate")

        status("evaluate-retrained-checkpoint")
        datasets = {split: instantiate(config, device="cuda") for split, config in cfg.datasets.items()}
        evaluate(reference, cfg, datasets, training / "checkpoint_best.pt", results / "retrained-checkpoint")
        status("completed", seconds=time.time() - started)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "train", "evaluate"), default="prepare")
    parser.add_argument("--started", type=float)
    parser.add_argument("--epochs", type=int, help="Training budget; defaults to the saved run or author configuration")
    main(parser.parse_args())
