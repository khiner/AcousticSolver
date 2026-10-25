#!/usr/bin/env python3
"""Compare matched coupled-wave trajectories with the pinned SeisSol solver."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external/seissol"
POINTS = [[0.13, 0.11, 0.12], [0.13, 0.11, -0.12]]


def exact(points, tag, time, scenario):
    source = (UPSTREAM / "src/Physics/Scenario/Scenarios.cpp").read_text()
    name = "ScholteWave" if scenario == "scholte" else "SnellsLaw"
    body = source.split(f"void seissol::physics::{name}::evaluateIn(")[1].split("template <typename RealT>")[
        0
    ]
    branches = body.split("if (isAcousticPart) {")[1].split("} else {")
    expressions = re.findall(
        r"dofsQp\(i, (\d)\)\s*=\s*([^;]+);", branches[0 if tag == (2 if scenario == "scholte" else 1) else 1]
    )
    x = np.asarray(points)
    env = {
        "x1": x[:, 0],
        "x3": x[:, 2],
        "t": time,
        "omega": 2 * np.pi,
        "pow": np.power,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
    }
    fields = np.zeros((len(x), 9))
    for index, expression in expressions:
        fields[:, int(index)] = eval(
            " ".join(expression.replace("std::", "").split()), {"__builtins__": {}}, env
        )
    return fields[
        :, [6, 7, 8, 0] if tag == (2 if scenario == "scholte" else 1) else [6, 7, 8, 0, 1, 2, 4, 5, 3]
    ]


def run(output, binary, cells, scenario):
    output = output / scenario
    fluid_tag = 2 if scenario == "scholte" else 1

    def field(points, tag, time):
        return exact(points, tag, time, scenario)

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for count in cells:
        case = output / f"n{count}"
        case.mkdir(exist_ok=True)
        (case / "output").mkdir(exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(UPSTREAM / "tests/Verification/minimesh.py"),
                "generate",
                "box48",
                "--cells",
                f"{count},{count},{count}",
                "--size",
                "1,1,1",
                "--origin=-.5,-.5,-.5",
                "--group-split",
                "z",
                "--bc",
                "all=analytical",
                "-o",
                str(case / "mesh.puml.h5"),
            ],
            check=True,
        )
        with h5py.File(case / "mesh.puml.h5") as h5:
            xyz = h5["geometry"][:]
            elements = []
            for vertices, tag in zip(h5["connect"][:], h5["group"][:]):
                if np.linalg.det((xyz[vertices[1:]] - xyz[vertices[0]]).T) < 0:
                    vertices[[1, 2]] = vertices[[2, 1]]
                elements.append({"tag": int(tag), "nodes": vertices.tolist()})
        mesh = {
            "geometry_degree": 1,
            "reference_nodes": [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
            "nodes": xyz.tolist(),
            "elements": elements,
        }
        (case / "mesh.json").write_text(json.dumps(mesh))
        text = (
            (UPSTREAM / "tests/Verification/cases/wave.par")
            .read_text()
            .replace("'Planarwave'", f"'{scenario}'")
            .replace("'material-elastic.yaml'", "'material.yaml'")
            .replace("2e-3", "0.02")
            .replace("FixTimeStep = 1e-4", "FixTimeStep = 5e-5")
        )
        (case / "parameters.par").write_text(text)
        materials = {1: (1, 1), 2: (0, 1)} if scenario == "scholte" else {1: (0, 1), 2: (4, 1)}
        yaml = "!Any\ncomponents:\n"
        native_materials = {}
        for tag, (mu, lam) in materials.items():
            yaml += f"  - !GroupFilter\n    groups: {tag}\n    components: !ConstantMap\n      map:\n        rho: 1\n        mu: {mu}\n        lambda: {lam}\n"
            native_materials[str(tag)] = (
                {"type": "fluid", "density": 1, "sound_speed": 1}
                if mu == 0
                else {
                    "type": "solid",
                    "density": 1,
                    "young_modulus": mu * (3 * lam + 2 * mu) / (lam + mu),
                    "poisson_ratio": lam / (2 * (lam + mu)),
                }
            )
        (case / "material.yaml").write_text(yaml)
        (case / "receiver.dat").write_text("\n".join(" ".join(map(str, p)) for p in POINTS) + "\n")
        with (case / "seissol.log").open("w") as log:
            subprocess.run(
                [str(binary.resolve()), "parameters.par"],
                cwd=case,
                env=dict(os.environ, OMP_NUM_THREADS="1"),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        config = {
            "mesh": "mesh.json",
            "degree": 2,
            "reference_seconds": 0.02,
            "sample_rate": 10000,
            "dt": 0.00005,
            "materials": native_materials,
            "receivers": [
                {"position": p, "field": f}
                for p in POINTS
                for f in ["vx", "vz", "pressure" if (p[2] > 0) == (fluid_tag == 2) else "szz"]
            ],
        }
        (case / "config.json").write_text(json.dumps(config))
        command = [
            str(ROOT / "build/ElasticReference"),
            str(case / "config.json"),
            str(case / "points.json"),
            str(case / "native.json"),
        ]
        subprocess.run(command, check=True)
        coordinates = json.loads((case / "points.json").read_text())
        data = {"omega": 2 * np.pi}
        for name, items in coordinates.items():
            data[name] = [
                {
                    "real": field(item["points"], item["tag"], 0).tolist(),
                    "imag": field(item["points"], item["tag"], 0.25).tolist(),
                }
                for item in items
            ]
        (case / "fields.json").write_text(json.dumps(data))
        subprocess.run(command + [str(case / "fields.json")], check=True)
        native = np.fromfile(case / "native.json.bin", dtype=np.float64).reshape(-1, 6)
        reference = []
        analytic = []
        for i, p in enumerate(POINTS):
            fluid = (i == 0) == (fluid_tag == 2)
            values = np.loadtxt(case / f"output/mini-receiver-{i+1:05d}.dat", skiprows=7)
            target = np.arange(1, len(native) + 1) / 10000
            wave = np.column_stack(
                [np.interp(target, values[:, 0], values[:, column]) for column in [7, 9, 1 if fluid else 3]]
            )
            if fluid:
                wave[:, 2] *= -1
            reference.append(wave)
            truth = np.array(
                [field([p], 2 if i == 0 else 1, t)[0][[0, 2, 3 if fluid else 5]] for t in target]
            )
            if fluid:
                truth[:, 2] *= -1
            analytic.append(truth)
        reference = np.hstack(reference)
        analytic = np.hstack(analytic)
        score = {
            "cells_per_axis": count,
            "native": json.loads((case / "native.json").read_text()),
            "receivers": [],
        }
        for i in range(2):
            part = slice(3 * i, 3 * i + 3)
            scale = np.linalg.norm(analytic[:, part])
            score["receivers"].append(
                {
                    "native_vs_seissol": float(np.linalg.norm(native[:, part] - reference[:, part]) / scale),
                    "native_vs_analytic": float(np.linalg.norm(native[:, part] - analytic[:, part]) / scale),
                    "seissol_vs_analytic": float(
                        np.linalg.norm(reference[:, part] - analytic[:, part]) / scale
                    ),
                }
            )
        reports.append(score)
        print(json.dumps(score, indent=2), flush=True)
    report = {
        "revision": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "results": reports,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for i in range(2):
        last = reports[-1]["receivers"][i]
        if max(last.values()) > 0.05:
            raise RuntimeError(f"{scenario} receiver accuracy failed: {last}")
        if (
            len(reports) > 1
            and last["native_vs_analytic"] >= reports[0]["receivers"][i]["native_vs_analytic"]
        ):
            raise RuntimeError(f"{scenario} refinement failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/elastic-reference/seissol")
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build/elastic-reference/seissol-build/seissol-elastic-o3-f64"
    )
    parser.add_argument("--cells", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--scenario", choices=["scholte", "snell"], default="scholte")
    args = parser.parse_args()
    run(args.output, args.binary, args.cells, args.scenario)
