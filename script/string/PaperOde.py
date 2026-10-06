#!/usr/bin/env python3
"""Reproduce Figures 1–2 of Russo, Ducceschi & Bilbao (2026), Sections 2–4.

Runs the complete published ODE sweep in native C++, then renders standalone PDF
and PNG figures. A second Verlet reference at twice the published rate measures
reference sensitivity. --smoke runs only a=1,2 and t=0.05 s for a bounded check.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = "https://nemusproject.eu/file/2026_russo_numerical.pdf"
DOI = "https://doi.org/10.1007/s11071-026-12708-0"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def potential(u, v, alpha, scheme):
    extension = (u * u + v * (2 + v)) / (np.sqrt(u * u + (1 + v) ** 2) + 1)
    if scheme == "RB":
        return (alpha - 1) * (.5 * u * u + .5 + v - extension)
    residual = .5 * (alpha - 1) * extension ** 2
    return residual + .5 * (u * u + v * v) if scheme == "V" else residual


def read_csv(path):
    with Path(path).open() as stream:
        return [{key: value if key == "scheme" else float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def figures(output, rows):
    u, v = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-.999, 1, 101))
    surfaces = {scheme: potential(u, v, 10000, scheme) for scheme in ("V", "RA", "RB")}
    np.savez_compressed(output / "figure1-data.npz", u=u, v=v, **surfaces)
    figure = plt.figure(figsize=(13, 4.4), layout="constrained")
    limit = max(float(np.max(values)) for values in surfaces.values())
    for index, (scheme, values) in enumerate(surfaces.items(), 1):
        axes = figure.add_subplot(1, 3, index, projection="3d")
        axes.plot_surface(u, v, values, cmap="viridis", rcount=51, ccount=51, linewidth=0, antialiased=True)
        axes.set(xlabel="$u$", ylabel="$v$", zlabel="$\\Phi$", title=scheme, zlim=(0, limit))
        axes.view_init(elev=27, azim=-125)
        axes.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))
    figure.suptitle(r"ODE SAV potentials, $\alpha=10000$ (equations 14 and 16)")
    for extension in ("png", "pdf"):
        figure.savefig(output / f"figure1-potentials.{extension}", dpi=180)
    plt.close(figure)

    times = sorted({row["time"] for row in rows})
    figure, grid = plt.subplots(len(times), 3, figsize=(12, 3.0 * len(times)), squeeze=False, layout="constrained")
    colors = {"V": "#2463ad", "RA": "#d46224", "RB": "#23864b"}
    for row_index, moment in enumerate(times):
        for column, field in enumerate(("u", "v", "psi")):
            axes = grid[row_index, column]
            for scheme in colors:
                for large in (False, True):
                    selected = sorted((row for row in rows if row["scheme"] == scheme and row["time"] == moment
                                       and (row["epsilon"] > 1) == large), key=lambda row: row["sample_rate"])
                    label = f"{scheme}, " + (r"$\epsilon=10^3$" if large else r"$\epsilon=\epsilon_{mach}$")
                    axes.loglog([row["sample_rate"] for row in selected],
                                [max(row[field + "_error"], 1e-18) for row in selected],
                                "--o" if large else "-o", color=colors[scheme], markersize=3, linewidth=1.2, label=label)
            axes.grid(True, which="both", linewidth=.4, alpha=.35)
            axes.set_title((r"$\psi(t_E-k/2)$" if field == "psi" else f"${field}(t_E)$") + f", $t_E={moment:g}$ s")
            axes.set_ylabel("absolute error")
            if row_index == len(times) - 1:
                axes.set_xlabel("sample rate (Hz)")
    handles, labels = grid[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=9)
    figure.suptitle(r"ODE convergence, $\alpha=10^5$, $w_0=(0.5,0.2)$, $\dot{w}_0=0$")
    for extension in ("png", "pdf"):
        figure.savefig(output / f"figure2-convergence.{extension}", dpi=180)
    plt.close(figure)


def main(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_suffix(".cpp")
    executable = output / "PaperOde"
    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise RuntimeError(f"Compiler not found: {args.compiler}")
    compile_command = [compiler, "-std=c++20", "-O3", "-ffp-contract=off", str(source), "-o", str(executable)]
    subprocess.run(compile_command, check=True)
    exponent, reference_exponent, duration = (2, 4, .05) if args.smoke else (8, 10, 5)
    command = [str(executable), "--output", str(output), "--max-exponent", str(exponent),
               "--reference-exponent", str(reference_exponent), "--duration", str(duration)]
    start = time.perf_counter()
    with (output / "run.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    audit = output / "reference-audit"
    audit_command = [str(executable), "--output", str(audit), "--max-exponent", str(exponent),
                     "--reference-exponent", str(reference_exponent + 1), "--duration", str(duration), "--reference-only"]
    subprocess.run(audit_command, check=True)
    elapsed = time.perf_counter() - start
    rows = read_csv(output / "results.csv")
    expected_count = 3 * 2 * exponent * (1 if args.smoke else 3)
    if len(rows) != expected_count or not all(np.isfinite(value) for row in rows for key, value in row.items() if key != "scheme"):
        raise RuntimeError("Incomplete or nonfinite ODE sweep")
    reference = json.loads((output / "reference.json").read_text())
    refined = {int(row["step"]): row for row in read_csv(audit / "reference.csv")}
    refined_rate = reference["sample_rate"] * 2
    for row in rows:
        full = refined[round(row["time"] * refined_rate)]
        half = refined[round(row["time"] * refined_rate) - refined_rate // (2 * int(row["sample_rate"]))]
        row["reference_refinement_u_delta"] = abs(row["reference_u"] - full["u"])
        row["reference_refinement_v_delta"] = abs(row["reference_v"] - full["v"])
        psi = np.sqrt(2 * potential(half["u"], half["v"], 1e5, row["scheme"]) + row["epsilon"])
        row["reference_refinement_psi_delta"] = abs(row["reference_psi"] - float(psi))
    checks = {
        "complete_finite_records": len(rows) == expected_count,
        "sav_discrete_energy": max(row["energy_max_relative_drift"] for row in rows) < 1e-8,
        "admissible_domain": min(row["min_v"] for row in rows) > -1,
        "reference_energy": reference["maximum_relative_energy_drift"] < 1e-5,
    }
    if not all(checks.values()):
        raise RuntimeError(f"ODE verification failed: {checks}")
    figures(output, rows)
    assumptions = [
        "Equation 23b prints B=2I+k^2(1-sigma)K; equation 21a and the restoring-force equation require the minus sign. The implementation follows equation 21a.",
        "The ODE section does not specify the discrete startup. Use w1=w0-k^2*grad(V(w0))/2 for zero initial velocity, and psi(1/2)=sqrt(2*Phi((w0+w1)/2)+epsilon).",
        "SAV equation 21 is solved through changes in position increments, algebraically equivalent to the rank-one position solve, avoiding cancellation in 2-k^2*K at MHz rates.",
        "Verlet uses kick-drift-kick velocity updates. The published a=10 reference defines all reported errors; a=11 is an additional reference-sensitivity audit.",
        "Psi reference is evaluated from the reference position exactly at tE-k/2 for each coarse rate, not at tE and not via temporal interpolation.",
        "Algebraic rationalizations of sqrt((1+v)^2+u^2)-1 and the Form B gradient avoid cancellation; formulas and dynamics are unchanged.",
        "Figure 1 uses caption alpha=10000, sampled u in [-2,2], v in [-0.999,1], and unscaled equation values. Its z-axis follows those values rather than inferring a hidden scaling from the printed illustration.",
        "This is a fresh implementation of the published ODE equations; the upstream MATLAB repository contains string solvers, not the ODE scripts. Exact digitized agreement with the printed plot is not asserted.",
    ]
    report = {
        "paper": DOI, "paper_pdf": PAPER, "complete_published_sweep": not args.smoke, "checks": checks, "pass": all(checks.values()),
        "parameters": {"alpha": 1e5, "initial_position": [.5, .2], "initial_velocity": [0, 0],
                       "base_sample_rate": 10000, "sav_exponents": list(range(1, exponent + 1)),
                       "reference_exponent": reference_exponent, "times": sorted({row["time"] for row in rows}),
                       "epsilon": [float(np.finfo(float).eps), 1000.]},
        "arithmetic": "IEEE binary64; floating-point contraction disabled", "assumptions": assumptions,
        "reference": reference, "reference_audit": json.loads((audit / "reference.json").read_text()),
        "maximum_sav_relative_energy_drift": max(row["energy_max_relative_drift"] for row in rows),
        "minimum_v": min(row["min_v"] for row in rows), "endpoint_records": len(rows), "results": rows,
        "simulation_wall_seconds": elapsed,
        "provenance": {"source_sha256": sha256(source), "analysis_sha256": sha256(__file__), "executable_sha256": sha256(executable),
                       "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
                       "compile_command": compile_command, "command": command, "reference_audit_command": audit_command,
                       "platform": platform.platform(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    paper = ROOT / "build/string-paper/paper.pdf"
    if paper.exists():
        report["provenance"]["paper_sha256"] = sha256(paper)
    report["artifact_sha256"] = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                                  if path.is_file() and path.name != "summary.json"}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(rows)} endpoint records; full published sweep={not args.smoke}; {elapsed:.2f}s -> {output}")
    print(f"Maximum SAV relative energy drift={report['maximum_sav_relative_energy_drift']:.3g}; min(v)={report['minimum_v']:.6g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "build/string-paper/ode"))
    parser.add_argument("--compiler", default="clang++")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
