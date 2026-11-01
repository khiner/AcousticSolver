"""Run the unchanged author trainer with durable per-epoch checkpoints and status."""

import json
import os
import runpy
import sys
import time
from pathlib import Path


def main():
    reference = Path(os.environ["STABLE_MODAL_REFERENCE"]).resolve()
    sys.path.insert(0, str(reference))
    from src.utils.trainer import Trainer
    from script.stable_modal_synthesis.cuda_graph import install

    install()
    original = Trainer._monitor_performance
    started = time.monotonic()

    def monitor(self, count):
        result = original(self, count)
        if int(os.environ["LOCAL_RANK"]) == 0:
            self._save_checkpoint()
            epoch = self._tracker["epoch"]
            report = dict(
                epoch=epoch,
                epochs=self._num_epochs,
                seconds=time.monotonic() - started,
                best_validation_loss=float(self._monitor_best),
                train={k: float(v) for k, v in self._tracker["train"].items()},
                valid={k: float(v) for k, v in self._tracker["valid"].items()},
            )
            path = self._save_dir / "progress.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(path)
            with (self._save_dir / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(report) + "\n")
            print(json.dumps(report), flush=True)
            # Retain the selected checkpoint and the two most recent complete epochs.
            for old in self._save_dir.glob("checkpoint_*.pt"):
                number = old.stem.removeprefix("checkpoint_")
                if number.isdigit() and int(number) < epoch - 1:
                    old.unlink()
        return result

    Trainer._monitor_performance = monitor
    os.chdir(reference)
    runpy.run_path(str(reference / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
