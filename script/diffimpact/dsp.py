"""DiffImpact synthesis on Apple Silicon. See NOTICE for upstream attribution."""

import math

import mlx.core as mx


def exp_sigmoid(x, exponent=10.0, maximum=2.0, minimum=1e-7):
    return maximum * mx.sigmoid(x) ** math.log(exponent) + minimum


def hann(size):
    # TensorFlow uses a symmetric window for odd lengths, periodic for even lengths.
    return 0.5 - 0.5 * mx.cos(2 * math.pi * mx.arange(size) / (size - size % 2))


def resample(x, size, window=False):
    count = x.shape[1]
    if window and (size % count or count + 1 >= size):
        raise ValueError("Hann upsampling requires an integer expansion greater than one")
    position = mx.arange(size, dtype=mx.float32) * (count / size)
    index = position.astype(mx.int32)
    fraction = position - index
    weight = (0.5 - 0.5 * mx.cos(math.pi * fraction)) if window else fraction
    weight = weight.reshape((1, size) + (1,) * (x.ndim - 2))
    return x[:, index] * (1 - weight) + x[:, mx.minimum(index + 1, count - 1)] * weight


def frames(x, size, hop):
    count = math.ceil(x.shape[-1] / hop)
    x = mx.pad(x, [(0, 0), (0, (count - 1) * hop + size - x.shape[-1])])
    return x[:, mx.arange(count)[:, None] * hop + mx.arange(size)[None, :]]


def convolve(audio, ir, delay=None):
    if ir.ndim == 2:
        ir = ir[:, None, :]
    count, length = ir.shape[1:]
    hop = math.ceil(audio.shape[1] / count)
    if math.ceil(audio.shape[1] / hop) != count:
        raise ValueError("Audio and filter frame counts do not match")
    size = 1 << (hop + length - 2).bit_length()
    blocks = mx.fft.irfft(mx.fft.rfft(frames(audio, hop, hop), n=size) * mx.fft.rfft(ir, n=size), n=size)
    segments = math.ceil(size / hop)
    blocks = mx.pad(blocks, [(0, 0), (0, 0), (0, segments * hop - size)])
    output = mx.zeros((audio.shape[0], (count + segments - 1) * hop))
    for i in range(segments):
        stripe = blocks[:, :, i * hop : (i + 1) * hop].reshape(audio.shape[0], -1)
        output = output + mx.pad(stripe, [(0, 0), (i * hop, (segments - i - 1) * hop)])
    start = (length - 1) // 2 - 1 if delay is None else delay
    return output[:, start : start + audio.shape[1]]


def filter_ir(magnitudes, window_size):
    ir = mx.fft.irfft(magnitudes.astype(mx.complex64))
    size = ir.shape[-1]
    window_size = min(window_size, size) if window_size > 0 else size
    window = hann(window_size)
    if window_size < size:
        half = (window_size + 1) // 2
        window = mx.concatenate((window[half:], mx.zeros((size - window_size,)), window[:half]))
        ir = ir * window
        # Preserve the released DDSP filter's asymmetric trimming convention.
        return mx.concatenate((ir[..., size - half + 2 :], ir[..., : half + 1]), axis=-1)
    return mx.roll(ir * mx.roll(window, size // 2), size // 2, axis=-1)


def filtered_noise(noise, magnitudes, window_size=257):
    return convolve(noise, filter_ir(magnitudes, window_size))


def modal_controls(gains, frequencies, dampings, hz_max):
    mel = mx.linspace(2595 * math.log10(1 + 10 / 700), 2595 * math.log10(1 + hz_max / 700), frequencies.shape[-1])
    centers = 700 * (10 ** (mel / 2595) - 1)
    frequencies = centers + (0.108 * centers + 24.7) * mx.tanh(frequencies)
    frequencies = mx.logaddexp(frequencies, 0) + 10 - mx.logaddexp(frequencies - (hz_max - 10), 0)
    return (mx.softmax(gains, axis=-1), frequencies, 10000 * exp_sigmoid(dampings - 1.5, exponent=4))


def modal_ir(gains, frequencies, dampings, size, rate, hz_max):
    gains, frequencies, dampings = modal_controls(gains, frequencies, dampings, hz_max)
    t = mx.arange(size, dtype=mx.float32)[None, :, None] / rate
    omega = frequencies * (2 * math.pi) / rate
    # Frequencies are constant: avoid float32 cumulative-sum drift and its activation storage.
    phase = omega * (mx.arange(size, dtype=mx.float32)[None, :, None] + 1)
    signal = mx.sum(mx.where(frequencies >= rate / 2, 0, gains) * mx.exp(-dampings * t) * mx.sin(phase), axis=-1)
    return mx.concatenate((mx.zeros_like(signal), signal), axis=1)


def gaussian(magnitudes, times, taus, size, rate):
    t = mx.arange(size, dtype=mx.float32)[None, :, None] / rate
    return mx.sum(magnitudes * mx.exp(-6 / taus**2 * (t - times - taus / 2) ** 2), axis=-1)


def contact(magnitudes, taus, size, rate, max_frequency=20):
    magnitudes = resample(magnitudes, size, window=True)[..., 0]
    taus = resample(taus, size, window=True)[..., 0]
    stride = int(rate / max_frequency)
    count = math.ceil(size / stride)
    padding = count * stride - size
    before = padding // 2
    padded = mx.pad(magnitudes, [(0, 0), (before, padding - before)], constant_values=-mx.inf)
    windows = padded.reshape(magnitudes.shape[0], count, stride)
    indices = mx.stop_gradient(mx.argmax(windows, axis=-1).astype(mx.int32) + mx.arange(count) * stride - before)
    peaks = mx.take_along_axis(magnitudes, indices, axis=1)
    neighbors = mx.clip(indices[..., None] + mx.array([-100, 0, 100]), 0, size - 1)
    weights = mx.take_along_axis(magnitudes[:, None, :], neighbors, axis=2)
    times = mx.sum(neighbors * weights, axis=-1) / mx.sum(weights, axis=-1) / rate
    widths = mx.take_along_axis(taus, indices, axis=1)
    return gaussian(peaks[:, None, :], times[:, None, :], widths[:, None, :], size, rate)


def reverb_ir(gains, decay, noise):
    t = mx.linspace(0, 1, noise.shape[1] // 50)[None, :, None]
    magnitudes = exp_sigmoid(gains - 3) * mx.exp(-(2 + mx.exp(decay + 4)) * t)
    ir = filtered_noise(noise, magnitudes)
    return mx.concatenate((mx.zeros_like(ir[:, :1]), ir[:, 1:]), axis=1)


def magnitude(audio, size, hop=None):
    return mx.abs(mx.fft.rfft(frames(audio, size, hop or size // 4) * hann(size)))


def safe_log(x):
    return mx.log(mx.maximum(x, 1e-5))


def spectral_loss(target, audio, sizes=(2048, 1024, 512, 256, 128), log_weight=1.0):
    loss = mx.array(0.0)
    for size in sizes:
        a, b = magnitude(target, size), magnitude(audio, size)
        loss = loss + mx.mean(mx.abs(a - b)) + log_weight * mx.mean(mx.abs(safe_log(a) - safe_log(b)))
    return loss
