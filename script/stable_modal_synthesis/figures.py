"""Render the paper's five comparisons using the released notebook plot functions."""

import ast
import json
from math import ceil, log, sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from librosa.display import specshow


def generate(reference, directory, dataset):
    from src.utils.misc import get_window

    source = (reference / "notebooks/test_nonlinear_string.py").read_text()
    source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
    tree = ast.parse(source)
    names = {"plot_wave", "plot_spec", "plot_displacement", "plot_displacement_grid", "Phi"}
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[]
    )
    ns = dict(
        torch=torch,
        np=np,
        plt=plt,
        mpl=matplotlib,
        sqrt=sqrt,
        log=log,
        ceil=ceil,
        get_window=get_window,
        specshow=specshow,
        num_modes=75,
        LOG_TICKS=[62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
        LOG_LABELS=["62.5", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"],
    )
    exec(compile(module, "author-plot-functions", "exec"), ns)
    selection = json.loads((directory / "figure-selection.json").read_text())
    output = directory / "figures"
    output.mkdir(exist_ok=True)

    def save(fig, name):
        for extension in ("pdf", "png"):
            fig.savefig(output / (name + "." + extension), dpi=160, bbox_inches="tight")
        plt.close(fig)

    worst = selection["worst"]
    target = dataset[worst]
    linear = dataset._getlin(worst)
    predicted = torch.load(directory / "worst.pt", weights_only=True)
    rate = target["fs"]
    frames = rate // 10
    q, tq, lq = predicted["output"][:75], target["target_output"][:75], linear["linear_output"][:75]
    w, tw, lw = predicted["w"], target["target_w"], linear["linear_w"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for axis, value, title in zip(axes, (tq, q, q), ("Target", "Predicted", "Relative absolute error")):
        ns["plot_displacement_grid"](
            value[:, :frames],
            rate,
            target_q=tq[:, :frames] if axis is axes[2] else None,
            num_points=1000,
            plot_bar=axis is axes[2],
            axs=axis,
        )
        axis.set_title(title)
    save(fig, "figure-1-displacement")
    errors = np.load(directory / "mode-mse.npz")
    fig, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    for name in ("linear", "predicted"):
        axis.semilogy(np.arange(1, 76), errors[name], label=name)
    axis.set(xlabel="Mode", ylabel="Mean squared displacement error (initial 100 ms)")
    axis.legend()
    save(fig, "figure-2-mode-errors")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, interval in zip(axes, ((0, 0.025), (0.505, 0.530))):
        ns["plot_wave"](tw, w, lw, rate, axs=axis)
        axis.set_xlim(*interval)
    save(fig, "figure-3-waveform")
    fig, axes = ns["plot_displacement"](tq, q, lq, rate, modes=[0, 29, 59])
    fig.set_size_inches(10, 7)
    for axis, mode in zip(np.asarray(axes).ravel(), (0, 29, 59)):
        start = target["exc_dur"]
        axis.set_xlim(start, start + 2 * np.pi * 20 / target["omega"][mode].item())
    save(fig, "figure-4-modal-trajectories")
    target = dataset[selection["strongest"]]
    predicted = torch.load(directory / "strongest.pt", weights_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, wave, title in zip(axes, (target["target_w"], predicted["w"]), ("Target", "Predicted")):
        ns["plot_spec"](wave, rate, freq_range=10000, plot_bar=axis is axes[1], axs=axis)
        axis.set_title(title)
    save(fig, "figure-5-spectrogram")
