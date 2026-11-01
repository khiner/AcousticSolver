"""Load the pinned author implementation without modifying its checkout."""

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "external/stable-modal-synthesis"
PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load(reference=DEFAULT_REFERENCE, threads=1):
    reference = Path(reference).resolve()
    for name, expected in PROTOCOL["sources"].items():
        if digest(reference / name) != expected:
            raise ValueError(f"Author source changed: {name}")
    sys.path.insert(0, str(reference))
    import src

    if Path(src.__file__).resolve() != reference / "src/__init__.py":
        raise RuntimeError("An unrelated src package was imported")
    torch.set_num_threads(threads)
    return OmegaConf.load(reference / PROTOCOL["config"])


def provenance(reference):
    return dict(
        protocol=PROTOCOL,
        python=platform.python_version(),
        platform=platform.platform(),
        device="cpu",
        dtype="float64",
        threads=torch.get_num_threads(),
        packages={
            name: importlib.metadata.version(name) for name in ("torch", "torch-dct", "numpy", "scipy", "hydra-core")
        },
        harness={p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        reference=str(Path(reference).resolve()),
    )


def metadata(config, device="cpu"):
    from src.generators import NonlinearString
    from src.utils.misc import gen_from_range

    generator = torch.Generator(device=device).manual_seed(config.seed)
    count = config.num_variations
    names = ("gamma", "kappa", "mu", "sigma0", "sigma1", "xe", "xo", "exc_amp", "exc_dur")
    values = {
        name: gen_from_range(config[name + "_range"], (count,), dtype=torch.float64, device=device, generator=generator)
        for name in names
    }
    omega, sigma = NonlinearString._calc_modes(
        values["gamma"], values["kappa"], values["sigma0"], values["sigma1"], config.num_modes
    )
    return [
        dict(
            idx=i,
            fs=config.fs,
            dur=config.dur,
            num_modes=config.num_modes,
            **{name: value[i].item() for name, value in values.items()},
            y0=torch.zeros(2 * config.num_modes, dtype=torch.float64),
            omega=omega[i].cpu(),
            sigma=sigma[i].cpu(),
            exc_st=0.0,
            exc_type=config.exc_type,
        )
        for i in range(count)
    ]


def generator(config, linear=False, duration=None):
    from src.generators import NonlinearString

    return NonlinearString(
        config.fs,
        duration if duration is not None else config.dur,
        config.method_name,
        config.num_modes,
        nl_name="zero" if linear else config.nl_name,
        method_kwargs=dict(config.method_kwargs),
        nl_kwargs=dict(config.nl_kwargs),
    ).double()


def generate(model, batch):
    names = ("y0", "gamma", "kappa", "mu", "sigma0", "sigma1", "xe", "xo", "exc_amp", "exc_dur", "exc_st", "exc_type")
    return model(**{key: batch[key] for key in names})


def learned(config, checkpoint=None):
    torch.manual_seed(config.seed)
    model = instantiate(config.model).double()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
    return model


def double_batch(batch):
    return {
        key: value.double() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in batch.items()
    }


def prepared_inputs(model, batch):
    fs, frames = model._calc_num_samples(batch["fs"], batch["dur"])
    phi = batch.get("phi_e")
    if phi is None:
        phi = model._Phi(batch["xe"], 75)
    excitation = batch.get("excitation")
    if excitation is None:
        excitation = model._calc_exc(
            fs,
            frames,
            batch["exc_amp"],
            batch["exc_dur"],
            batch["exc_st"],
            batch["exc_type"],
            device=batch["y0"].device,
        )
    return fs, frames, phi, excitation


def pickup(model, batch, output):
    phi = batch.get("phi_o")
    if phi is None:
        phi = model._Phi(batch["xo"], 75)
    return (phi.to(output)[:, :, None] * output[:, :75]).sum(1)


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    return value
