"""The released fitting experiments and the Appendix C audio encoder."""

import math

import mlx.core as mx
import mlx.nn as nn

from . import dsp


def mfcc(audio, rate):
    frequencies = mx.linspace(0, rate / 2, 513)
    mel = lambda x: 1127 * mx.log(1 + x / 700)
    edges = mx.linspace(mel(mx.array(4.0)), mel(mx.array(min(16000.0, rate / 2))), 130)
    slopes = (
        (mel(frequencies)[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2]),
        (edges[2:] - mel(frequencies)[:, None]) / (edges[2:] - edges[1:-1]),
    )
    weights = mx.maximum(0, mx.minimum(*slopes))
    weights = mx.concatenate((mx.zeros_like(weights[:1]), weights[1:]))
    logmel = dsp.safe_log(dsp.magnitude(audio, 1024) @ weights)
    dct = math.sqrt(2 / 128) * mx.cos(math.pi / 128 * (mx.arange(128)[:, None] + 0.5) * mx.arange(40))
    return logmel @ dct


class GRU(nn.Module):
    """Keras reset-after GRU, retaining both bias rows as one trainable variable."""

    def __init__(self, inputs, channels):
        super().__init__()
        scale = 1 / math.sqrt(channels)
        self.Wx = mx.random.uniform(-scale, scale, (3 * channels, inputs))
        self.Wh = mx.random.uniform(-scale, scale, (3 * channels, channels))
        self.bias = mx.zeros((2, 3 * channels))

    def __call__(self, x):
        x = mx.addmm(self.bias[0], x, self.Wx.T)
        hidden = mx.zeros((*x.shape[:-2], self.Wh.shape[1]))
        states = []
        for i in range(x.shape[-2]):
            xz, xr, xn = mx.split(x[..., i, :], 3, axis=-1)
            hz, hr, hn = mx.split(mx.addmm(self.bias[1], hidden, self.Wh.T), 3, axis=-1)
            z, r = mx.sigmoid(xz + hz), mx.sigmoid(xr + hr)
            hidden = z * hidden + (1 - z) * mx.tanh(xn + r * hn)
            states.append(hidden)
        return mx.stack(states, axis=-2)


class Encoder(nn.Module):
    """Released MFCC encoder and RnnFcDecoder with explicit single-latent wiring."""

    def __init__(self, channels=512):
        super().__init__()
        self.scale = mx.ones((1, 1, 40))
        self.shift = mx.zeros((1, 1, 40))
        self.input_gru = GRU(40, channels)
        self.conv = nn.Conv1d(channels, channels, 7, padding=3)
        self.projection = nn.Linear(channels, 32)
        self.input_fc = self.stack(32, channels)
        self.output_gru = GRU(channels, channels)
        self.output_fc = self.stack(2 * channels, channels)
        self.head = nn.Linear(channels, 2)

    @staticmethod
    def stack(inputs, channels):
        return nn.Sequential(
            *[
                layer
                for i in range(3)
                for layer in (
                    nn.Linear(inputs if i == 0 else channels, channels),
                    nn.LayerNorm(channels, eps=0.001),
                    nn.LeakyReLU(0.2),
                )
            ]
        )

    def __call__(self, features, frames):
        mean = mx.mean(features, axis=1, keepdims=True)
        variance = mx.mean((features - mean) ** 2, axis=1, keepdims=True)
        z = (features - mean) * mx.rsqrt(variance + 1e-5) * self.scale + self.shift
        z = self.projection(nn.leaky_relu(self.conv(self.input_gru(z)), 0.2))
        z = self.input_fc(dsp.resample(z, frames))
        z = mx.concatenate((z, self.output_gru(z)), axis=-1)
        return self.head(self.output_fc(z))


class Scene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = dict(config)
        self._rate = config["rate"] * 2
        self._size = config["samples"] * 2
        count = config.get("recordings", config["bodies"])
        modes = config["modes"]
        for name in ("frequencies", "gains", "dampings"):
            self[name] = mx.random.normal((config["bodies"], 1, modes))
        self.noise_magnitudes = mx.zeros((count, 200))
        self.reverb_gains = mx.full((count, 1, 32), -1.0)
        self.reverb_decay = mx.full((count, 1, 32), 3.0)
        self.acceleration_scale = mx.full((count, 1), 0.2)

    def noise(self, batch, key):
        keys = mx.random.split(key, 3)
        return {
            "contact": mx.random.normal((batch, self._config["frames"], 1), key=keys[0]),
            "background": mx.random.uniform(-1, 1, (batch, self._size), key=keys[1]),
            "reverb": mx.random.uniform(-1, 1, (batch, self._config["reverb_samples"]), key=keys[2]),
        }

    def body_ir(self):
        c = self._config
        return dsp.modal_ir(self.gains, self.frequencies, self.dampings, c["ir_samples"], self._rate, c["hz_max"])

    def render_contact(self, magnitudes, durations, draws):
        return self.render(dsp.contact(magnitudes, durations, self._size, self._rate), draws)

    def render(self, force, draws):
        body = dsp.convolve(force, self.body_ir())
        acceleration = force * self.acceleration_scale
        noise = dsp.filtered_noise(draws["background"], dsp.exp_sigmoid(self.noise_magnitudes - 4))
        dry = body + acceleration + noise
        room = dsp.reverb_ir(self.reverb_gains, self.reverb_decay, draws["reverb"])
        wet = dsp.convolve(dry, room, delay=0)
        return {
            "audio": (dry + wet)[:, ::2],
            "body": body[:, ::2],
            "acceleration": acceleration[:, ::2],
            "noise": noise[:, ::2],
            "reverb": wet[:, ::2],
            "force": force,
        }


class Hammer(Scene):
    def __init__(self, config):
        super().__init__(config)
        shape = (config.get("recordings", config["bodies"]), config["frames"], 1)
        self.mags = mx.zeros(shape)
        self.taus = mx.zeros(shape)
        self.stdevs = mx.full(shape, 0.1)
        self.freeze(keys=["reverb_gains", "reverb_decay"])

    def __call__(self, batch, draws):
        mags = dsp.exp_sigmoid(self.mags + mx.abs(self.stdevs) * draws["contact"] - 1.5)
        taus = dsp.exp_sigmoid(self.taus, 4, 0.003, 3 / self._rate)
        return self.render_contact(mags, taus, draws)


class Autoencoder(Scene):
    def __init__(self, config):
        super().__init__(config)
        if config["bodies"] != 1:
            raise ValueError("The paper trains a separate autoencoder per object")
        self.encoder = Encoder(config["channels"])
        self.tau_bias = mx.zeros((1, 1, 1))

    def controls(self, features):
        controls = self.encoder(features, self._config["frames"])
        mags = dsp.exp_sigmoid(controls[..., :1] - 1.5)
        taus = dsp.exp_sigmoid(controls[..., 1:] + self.tau_bias, 4, 0.003, 3 / self._rate)
        return mags, taus

    def __call__(self, batch, draws):
        return self.render_contact(*self.controls(batch["mfcc"]), draws)


class Separation(Scene):
    def __init__(self, config):
        super().__init__(config)
        self.dampings = self.dampings - 1
        for name in ("frequencies", "gains", "dampings"):
            self["tool_" + name] = mx.random.normal((config["tools"], 1, config["modes"] // 2))
        self.tool_dampings = self.tool_dampings - 1
        self.noise_magnitudes = mx.zeros((1, 100))
        self.noise_scale = mx.full((config["recordings"], 1), 0.1)
        self.acceleration_scale = mx.full((config["recordings"], 1), 0.1)
        self.mag = mx.full((config["recordings"], 1, 1), 0.1)
        self.tau = mx.full((config["recordings"], 1, 1), 0.1)
        self.peak_time = mx.array(config["peak_times"])[:, None, None]
        self.reverb_gains = mx.ones((config["tools"], 1, 16))
        self.reverb_decay = -mx.ones((config["tools"], 1, 16))

    def __call__(self, batch, draws):
        ids, objects, tools = batch["indices"], batch["objects"], batch["tools"]
        force = dsp.gaussian(
            self.mag[ids], 0.01 * self.peak_time[ids], 0.001 * self.tau[ids] + 3 / self._rate, self._size, self._rate
        )
        body = dsp.convolve(force, self.body_ir()[objects])
        c = self._config
        ir = dsp.modal_ir(
            self.tool_gains, self.tool_frequencies, self.tool_dampings, c["ir_samples"], self._rate, c["hz_max"]
        )
        tool = dsp.convolve(force, ir[tools])
        acceleration = force * (self.acceleration_scale[ids] + 1e-7)
        noise = self.noise_scale[ids] * dsp.filtered_noise(
            draws["background"], dsp.exp_sigmoid(self.noise_magnitudes - 3.5), 64
        )
        room = dsp.reverb_ir(self.reverb_gains[tools], self.reverb_decay[tools], draws["reverb"])
        dry = body + tool + acceleration + noise
        wet = dsp.convolve(dry, room, delay=0)
        separated_tool = tool + acceleration + dsp.convolve(tool + acceleration, room, delay=0)
        return {
            "audio": (dry + wet)[:, ::2],
            "body": body[:, ::2],
            "tool": tool[:, ::2],
            "separated_tool": separated_tool[:, ::2],
            "acceleration": acceleration[:, ::2],
            "noise": noise[:, ::2],
            "reverb": wet[:, ::2],
            "force": force,
        }


MODELS = {"hammer": Hammer, "separation": Separation, "autoencoder": Autoencoder}
