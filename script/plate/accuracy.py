"""Shared plate capture metrics and provenance; no solver implementation."""
import hashlib
import json
from pathlib import Path
import subprocess

from .env import ROOT

DIGITS = (64, 128)  # 214 and 427 binary bits.
TOLERANCE = 1e-8
NFFT, HOP = 2048, 512


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execute(command, log):
    print(' '.join(str(x) for x in command), flush=True)
    with log.open('w') as stream:
        result = subprocess.run([str(x) for x in command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')


def verify_artifacts(directory, expected):
    actual = {name: digest(directory / name) for name in expected}
    if actual != expected:
        raise ValueError(f'Changed output artifacts: {directory}')
    return actual


def identity(paths):
    # Required provenance inputs must exist; never silently omit a missing file.
    return {str(Path(p).resolve()): digest(p) for p in paths}


def native_paths():
    return [ROOT / 'CMakeLists.txt', ROOT / 'build/PlateSolve',
            ROOT / 'build/libPlateCore.a', ROOT / 'build/libAcousticCore.a',
            *sorted((ROOT / 'src/Plate').glob('*')),
            ROOT / 'script/plate/accuracy.py', ROOT / 'script/plate/env.py',
            Path('/opt/homebrew/lib/libmpfr.dylib'), Path('/opt/homebrew/lib/libgmp.dylib')]


def compare(a, b, count, modes, channels):
    import numpy as np

    def load(prefix, suffix, columns):
        path = prefix.with_suffix(suffix)
        if path.stat().st_size != count * columns * 8:
            raise ValueError(f'Output shape mismatch: {path}')
        values = np.memmap(path, dtype='<f8', mode='r', shape=(count, columns))
        return values

    def relative(x, y, scale=None):
        return float(np.linalg.norm(x-y) / max(float(np.linalg.norm(y if scale is None else scale)), 1e-300))

    ma, mb = [json.loads(p.with_suffix('.json').read_text()) for p in (a, b)]
    sample_rate = mb['config']['sample_rate']
    block_size = max(1, round(sample_rate * .01))
    qa, qb = load(a, '.states.bin', modes), load(b, '.states.bin', modes)
    error, norm, blocks = 0., 0., []
    for start in range(0, count, block_size):
        x, y = qa[start:start+block_size], qb[start:start+block_size]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite modal state')
        e, n = float(np.sum((x-y)**2)), float(np.sum(y*y))
        error += e
        norm += n
        blocks.append(dict(start_seconds=start/sample_rate, relative_l2=(e/max(n, 1e-300))**.5))
    da, db = load(a, '.diagnostics.bin', 6), load(b, '.diagnostics.bin', 6)
    wa, wb = load(a, '.bin', channels), load(b, '.bin', channels)
    if not all(np.isfinite(v).all() for v in (da, db, wa, wb)):
        raise ValueError('Nonfinite waveform or diagnostics')
    metrics = dict(waveform=relative(wa, wb), modal_state=(error/max(norm, 1e-300))**.5,
                   max_modal_block=max(v['relative_l2'] for v in blocks))
    for i, field in enumerate(('energy', 'input_power', 'dissipated_power', 'psi', 'half_potential', 'drift')):
        metrics[field] = relative(da[:, i], db[:, i], db[:, 3] if i == 5 else None)
    for field in ('q', 'previous', 'psi'):
        metrics[f'final_{field}'] = relative(np.asarray(ma['final_state'][field]), np.asarray(mb['final_state'][field]))
    return dict(metrics=metrics, modal_blocks=blocks)


def features(prefix, config):
    import numpy as np
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    if metadata["config"] != config:
        raise ValueError(f"Configuration mismatch: {prefix}")
    fs, channels = config["sample_rate"], len(config["pickups"])
    count = int(fs * config["seconds"]) * config.get("repeat_count", 1)
    wave = np.fromfile(prefix.with_suffix(".bin"), dtype="<f8").reshape(count, channels)
    diagnostics = np.fromfile(prefix.with_suffix(".diagnostics.bin"), dtype="<f8").reshape(count, 6)
    if not np.isfinite(wave).all() or not np.isfinite(diagnostics).all() or count < NFFT:
        raise ValueError(f"Invalid samples: {prefix}")
    starts = np.arange(0, count, round(.05 * fs))
    lengths = np.diff(np.append(starts, count))
    rms = np.sqrt(np.add.reduceat(wave * wave, starts, axis=0) / lengths[:, None])
    frames = np.lib.stride_tricks.sliding_window_view(wave.T, NFFT, axis=1)[:, ::HOP, :]
    window = np.hanning(NFFT)
    power = (np.abs(np.fft.rfft(frames * window, axis=2)) ** 2 / (NFFT * np.sum(window ** 2))).transpose(1, 0, 2)
    power[:, :, 1:-1] *= 2
    hz = np.fft.rfftfreq(NFFT, 1 / fs)
    edges = [0, *[f for f in (125, 250, 500, 1000, 2000, 4000, 8000, 17000) if f < fs / 2], fs / 2]
    masks = [(hz >= lo) & ((hz < hi) if hi != fs / 2 else (hz <= hi)) for lo, hi in zip(edges[:-1], edges[1:])]
    bands = np.stack([power[:, :, mask].sum(axis=2) for mask in masks], axis=2)
    return dict(wave=wave, energy=diagnostics[:, 0], rms=rms,
                rms_time=(starts + lengths / 2) / fs, bands=bands,
                spectrum=power.mean(axis=0), magnitude=np.sqrt(power),
                band_edges=edges, hz=hz, time=(np.arange(len(power)) * HOP + (NFFT - 1) / 2) / fs)


def energy_work_balance(prefix, sample_rate):
    import numpy as np
    diagnostics = np.fromfile(prefix.with_suffix('.diagnostics.bin'), dtype='<f8').reshape(-1, 6)
    if not len(diagnostics) or not np.isfinite(diagnostics).all():
        raise ValueError(f'Invalid diagnostics: {prefix}')
    energy = diagnostics[:, 0]
    work = np.concatenate(([0.], np.cumsum(diagnostics[:-1, 1] - diagnostics[:-1, 2]) / sample_rate))
    return float(np.max(np.abs(energy - energy[0] - work)) / max(np.max(np.abs(energy)), 1e-300))


def compare_acoustics(actual, reference):
    import numpy as np

    def relative(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

    a, b = actual["bands"].mean(axis=0), reference["bands"].mean(axis=0)
    floor = max(float(b.sum()) * 1e-12, 1e-300)
    band_db = 10 * np.log10(np.maximum(a, floor) / np.maximum(b, floor))
    return dict(waveform_relative_l2=relative(actual["wave"], reference["wave"]),
                rms_envelope_relative_l2=relative(actual["rms"], reference["rms"]),
                band_evolution_relative_l2=relative(np.sqrt(actual["bands"]), np.sqrt(reference["bands"])),
                stft_magnitude_relative_l2=relative(actual["magnitude"], reference["magnitude"]),
                mean_spectrum_relative_l2=relative(np.sqrt(actual["spectrum"]), np.sqrt(reference["spectrum"])),
                band_power_relative_l1=float(np.abs(a - b).sum() / max(b.sum(), 1e-300)),
                max_abs_band_db=float(np.max(np.abs(band_db))), band_db=band_db.tolist(),
                energy_relative_l2=relative(actual["energy"], reference["energy"]))
