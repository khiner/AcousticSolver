"""Released oscillator, filtered noise, and spectral/transport objectives."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from geomloss import SamplesLoss

from . import DEVICE


def modified_sigmoid(x):
    return 2 * x.sigmoid().pow(2.3) + 1e-6


class Weighted(nn.Module):
    def __init__(self, values, shape=(), spread=1.0, device=DEVICE):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32, device=device))
        self.raw = nn.Parameter(torch.empty(*shape, len(values), device=device).uniform_(-spread, spread))

    def forward(self):
        weights = F.softplus(self.raw)
        return (weights * self.values).sum(-1) / weights.sum(-1)


def render(frequencies, damping, amplitudes, force, samples=8000, rate=32000):
    frequencies = frequencies.reshape(1, -1, 1)
    damping = damping.reshape(1, -1, 1)
    damped = ((2 * math.pi * frequencies).square() - damping.square()).sqrt() / (2 * math.pi)
    # The release integrates frequency and decay before producing the first sample.
    time = torch.arange(1, samples + 1, dtype=frequencies.dtype, device=frequencies.device)
    phase = (damped / rate) * time
    decay = (damping / rate) * time
    signal = (amplitudes * (-decay).exp() * (2 * math.pi * phase).sin()).sum(1)
    signal = signal.expand(len(force), -1)
    output = F.conv1d(signal[None], force[:, None].flip(-1), groups=len(force), padding=force.shape[-1] - 1)
    return output[0, :, :samples], damped.expand(len(force), -1, samples)


class FilteredNoise(nn.Module):
    def __init__(self, recordings, samples, bins=65, frame=64):
        super().__init__()
        self.samples, self.frame, self.bins = samples, frame, bins
        self.raw = nn.Parameter(torch.empty(recordings, samples // frame + 1, bins, device=DEVICE).uniform_(-1, 1))

    def forward(self, noise=None):
        b, frames, bins = self.raw.shape
        ir_size = 2 * bins - 1
        ir = torch.fft.irfft(modified_sigmoid(self.raw), n=ir_size)
        ir = ir.roll(bins - 1, dims=-1) * torch.hann_window(ir_size, device=DEVICE)
        if noise is None:
            # Draw on CPU, as upstream does, to make the saved RNG state portable.
            noise = (torch.rand(b, frames, self.frame) * 2 - 1).to(DEVICE)
        length = ir_size + self.frame - 1
        filtered = torch.fft.irfft(torch.fft.rfft(ir, n=length) * torch.fft.rfft(noise, n=length), n=length)
        kernel = torch.eye(length, device=DEVICE)[:, None]
        return F.conv_transpose1d(filtered.transpose(1, 2), kernel, stride=self.frame)[:, 0, : self.samples]


class ModalFit(nn.Module):
    """The free-modal damping-curve estimator used before real-audio material fitting."""

    def __init__(self, recordings, modes, samples, material):
        super().__init__()
        self.frequency = Weighted([20, 16000], (1, modes, 1), 4)
        self.alpha = Weighted(
            torch.logspace(math.log10(material[3] / 10), math.log10(material[3] * 100), 64), (1, modes, 1), 4
        )
        self.beta = Weighted(
            torch.logspace(math.log10(material[4] / 10), math.log10(material[4] * 100), 64), (1, modes, 1), 4
        )
        self.amplitude = nn.Parameter(torch.empty(recordings, modes, 1, device=DEVICE).uniform_(0, 0.04))
        self.noise = FilteredNoise(recordings, samples)
        self.samples = samples

    def controls(self):
        f = self.frequency()
        return f, 0.5 * (self.alpha() + self.beta() * (2 * math.pi * f).square())

    def forward(self, force, rate=32000, noise=None):
        f, d = self.controls()
        signal, _ = render(f, d, modified_sigmoid(self.amplitude), force, self.samples, rate)
        return signal + 5e-4 * self.noise(noise)


def spectral_points(spectrum, frequency=None, rate=32000):
    bins = spectrum.shape[-2]
    features = F.interpolate(spectrum.detach(), size=3, mode="linear")
    positions = (
        (torch.arange(bins, device=spectrum.device, dtype=torch.float32) / bins).expand(len(spectrum), -1).clone()
    )
    if frequency is not None:
        locations = (bins / (rate // 2) * frequency).reshape(-1)
        for width in (2, 1, 0):
            for sign in (-1, 1):
                p = locations + sign * width
                indices = p.long()
                valid = (indices >= 0) & (indices < bins)
                positions[:, indices[valid]] = p[valid] / bins
    return torch.cat((features, positions[..., None]), dim=-1)


class SpectralLoss(nn.Module):
    def __init__(self, sizes, kind="l1", rate=32000):
        super().__init__()
        self.sizes, self.kind, self.rate = sizes, kind, rate
        self.transport = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, backend="tensorized")

    @staticmethod
    def spectrum(audio, size):
        return (
            torch.stft(audio, size, size // 4, window=torch.hann_window(size, device=audio.device), return_complex=True)
            .abs()
            .square()
        )

    def forward(self, prediction, target, frequency=None, scale=1.0):
        if self.kind == "transport":
            prediction = prediction / (prediction.detach().amax(-1, keepdim=True) + 1e-7)
            target = target / (target.detach().amax(-1, keepdim=True) + 1e-7)
        total = 0
        for size in self.sizes:
            a, b = self.spectrum(prediction, size), self.spectrum(target, size)
            la, lb = (a + 1e-7).log2(), (b + 1e-7).log2()
            if self.kind == "transport":
                bins = int(a.shape[-2] * scale)
                pa, pb = (la[:, :bins] - math.log2(1e-7)) / 40, (lb[:, :bins] - math.log2(1e-7)) / 40
                total = total + self.transport(spectral_points(pa, frequency, self.rate), spectral_points(pb)).sum()
                total = total + self.transport(spectral_points(a, frequency, self.rate), spectral_points(b)).sum()
            elif self.kind == "l1":
                w = 1 - torch.linspace(1, 0.9, a.shape[-1], device=a.device)
                w = w / w.sum() * len(w)
                total = total + ((a - b)[:, 1:] * w).abs().mean() + ((la - lb)[:, 1:] * w).abs().mean()
            elif self.kind == "rmse":
                bins = int(a.shape[-2] * scale)
                total = total + (la[:, :bins] - lb[:, :bins]).square().mean().sqrt()
            else:
                raise ValueError(self.kind)
        return total
