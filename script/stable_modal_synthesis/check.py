"""Independent equation, gradient, energy, source-audio, and execution checks."""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import default_collate

from . import reference as ref


def run(reference, output, threads):
    cfg = ref.load(reference, threads)
    from src.ode.methods.sav import SAV
    from src.ode.problem import InitialValueProblem
    from src.datasets import SliceCollator
    from src.losses import MSEPosVelLoss

    report = dict(provenance=ref.provenance(reference), passed=False)
    model = ref.learned(cfg, reference / ref.PROTOCOL["checkpoint"])
    nl = model.nl
    torch.manual_seed(71)
    q = (torch.randn(2, 75, dtype=torch.float64) * 1e-4).requires_grad_()
    (grad,) = torch.autograd.grad(nl.potential(q).sum(), q)
    error = float((grad + nl(q)).abs().max().detach())
    assert error < 1e-10
    assert (nl.potential(q) >= 0).all()
    report["force_negative_potential_gradient_max_error"] = error

    method = SAV(nl, lambda0=1000.0)
    meta = ref.metadata(cfg.datasets.test)
    batch = default_collate(meta[:2])
    p = torch.randn_like(q) * 0.01
    initial = torch.cat((q.detach(), p), dim=-1)
    ivp = InitialValueProblem(
        96000,
        2,
        initial,
        batch["omega"],
        batch["sigma"],
        batch["mu"],
        model._Phi(batch["xe"], 75),
        torch.zeros(2, 2, dtype=torch.float64),
    )
    y, params = method.init(ivp)
    with torch.no_grad():
        actual = method.step(0, y, params)
        k = params.k
        q0, p0, psi = torch.tensor_split(y, (75, 150), dim=-1)
        half = q0 + 0.5 * k * p0
        g = method._calc_g_std(half) + method._calc_g_mod(q0, p0, psi)
        expected = []
        for i in range(2):
            a = 0.25 * k * k * float(batch["mu"][i]) ** 2 * np.outer(g[i], g[i])
            lhs = np.diag(params.dp[i]) + a
            rhs = (np.diag(params.dm[i]) - a) @ p0[i].numpy()
            rhs += k * (
                -params.omega_sq[i].numpy() * half[i].numpy()
                - float(batch["mu"][i]) ** 2 * g[i].numpy() * float(psi[i, 0])
            )
            next_p = np.linalg.solve(lhs, rhs)
            next_q = half[i].numpy() + 0.5 * k * next_p
            next_psi = psi[i].numpy() + k * np.dot(g[i], 0.5 * (p0[i].numpy() + next_p))
            expected.append(np.r_[next_q, next_p, next_psi])
        error = float(np.max(np.abs(actual.numpy() - np.array(expected))))
        assert error < 1e-11
        report["sav_dense_solve_max_error"] = error

    # The released training rule intentionally detaches the drift-control term.
    correction = method._calc_g_mod(q, p, y[:, -1:])
    assert not correction.requires_grad
    method._lambda0 = 0.0
    scalar = method.step(0, y.detach(), params)[:, :75].square().sum()
    weight = nl._weight
    (gradient,) = torch.autograd.grad(scalar, weight)
    index = np.unravel_index(int(gradient.abs().argmax()), gradient.shape)
    original = float(weight[index].detach())
    epsilon = 1e-5
    with torch.no_grad():
        weight[index] = original + epsilon
        plus = method.step(0, y.detach(), params)[:, :75].square().sum().item()
        weight[index] = original - epsilon
        minus = method.step(0, y.detach(), params)[:, :75].square().sum().item()
        weight[index] = original
    fd = (plus - minus) / (2 * epsilon)
    ad = gradient[index].item()
    relative = abs(ad - fd) / max(abs(ad), abs(fd), 1e-20)
    assert relative < 1e-4
    report["sav_weight_gradient"] = dict(
        ad=ad, finite_difference=fd, relative_error=relative, drift_control_disabled_for_finite_difference=True
    )

    method._lambda0 = 1000.0
    ivp.sigma = torch.zeros_like(ivp.sigma)

    def energy(state):
        q, p, psi = torch.tensor_split(state, (75, 150), dim=-1)
        return (
            0.5 * (p.square() + ivp.omega.square() * (q.square() - 0.25 * params.k**2 * p.square())).sum(-1)
            + 0.5 * ivp.mu.square() * psi[:, 0].square()
        )

    with torch.no_grad():
        y, params = method.init(ivp)
        e0 = energy(y)
        maximum = 0.0
        for _ in range(1000):
            y = method.step(0, y, params)
            maximum = max(maximum, float(((energy(y) - e0) / e0).abs().max()))
    assert maximum < 1e-10
    report["unforced_undamped_energy_relative_drift_1000_steps"] = maximum

    batch = default_collate(meta[:1])
    start = time.perf_counter()
    target = ref.generate(ref.generator(cfg.datasets.test, duration=0.1), batch)
    batch["dur"] = torch.tensor([0.1], dtype=torch.float64)
    with torch.no_grad():
        pred = model(**batch)
    wav = reference / "audio" / ref.PROTOCOL["audio_directories"]["test"] / "0.wav"
    published, rate = sf.read(wav)
    generated = target["w"][0].numpy()
    generated /= np.max(np.abs(generated))
    prefix = published[: len(generated)]
    audio_error = float(np.linalg.norm(generated - prefix) / np.linalg.norm(prefix))
    report["published_target_audio_check"] = dict(
        index=0,
        seconds=0.1,
        rate=rate,
        relative_error=audio_error,
        audio_sha256=ref.digest(wav),
        normalization="generated prefix peak; published file is independently peak-normalized",
    )
    report["reference_100ms_seconds"] = time.perf_counter() - start

    instance = dict(meta[0], target_output=target["output"][0], target_w=target["w"][0], dur=0.1)
    sliced = ref.double_batch(SliceCollator(0.001)([instance]))
    sliced = {key: value[:32] if isinstance(value, torch.Tensor) else value for key, value in sliced.items()}
    train_model = ref.learned(cfg)
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.001)
    started = time.perf_counter()
    prediction = train_model(**sliced)
    loss = MSEPosVelLoss()({**sliced, **prediction})["loss"]
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in train_model.parameters())
    optimizer.step()
    report["training_step"] = dict(
        segments=32,
        samples=96,
        loss=loss.item(),
        seconds=time.perf_counter() - started,
        scope="execution and finite gradients; not trained-quality evidence",
    )
    report["passed"] = True
    ref.write_json(output, report)
    print({k: v for k, v in report.items() if k != "provenance"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ref.DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    run(args.reference.resolve(), args.output, args.threads)
