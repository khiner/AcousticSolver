"""Evaluate the Metal acoustic objective on the current height texture."""

import json
from pathlib import Path
import subprocess
import numpy as np


def evaluate(problem, pixels, input_path, output_path, executable, sample_only=False):
    height, width = pixels.shape
    input_path.write_text(
        json.dumps(
            dict(
                problem=problem,
                design=dict(width=width, height=height, initial_pixels=pixels.ravel().tolist()),
            )
        )
        + "\n"
    )
    command = [str(executable.resolve()), str(input_path), str(output_path)]
    if sample_only:
        command.append("--sample-only")
    subprocess.run(command, check=True)
    return json.loads(output_path.read_text())


class Acoustic:
    def __init__(self, problem, output, executable=Path("build/BemTextureSolve")):
        self.problem = problem
        self.output = output
        self.executable = executable.resolve()
        output.mkdir(parents=True, exist_ok=True)

    def gradient(self, pixels, frequency):
        height, width = pixels.shape
        result = evaluate(
            dict(self.problem, frequency_hz=frequency),
            pixels,
            self.output / "input.json",
            self.output / "result.json",
            self.executable,
        )
        gradient = np.asarray(result["gradient"], dtype=np.float32).reshape(
            height, width
        )
        if not np.isfinite(gradient).all() or not np.isfinite(result["loss"]):
            raise ValueError("Nonfinite acoustic objective")
        return result["loss"], gradient, result
