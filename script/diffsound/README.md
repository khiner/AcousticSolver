# DiffSound

Apple Silicon port of [DiffSound: Differentiable Modal Sound Rendering and Inverse Rendering for Diverse Inference Tasks](https://jiajunwu.com/papers/diffsound_siggraph.pdf) (SIGGRAPH 2024), based on [the author release](https://github.com/TechnetiumMan/DiffSound/tree/3a0be14b8bfa95dcbcd98f3c233e981c713d021b).

The port includes linear/quadratic tetrahedral FEM, generalized eigenvalue derivatives, damped synthesis and filtered noise, spectral and optimal-transport losses, material/damping inference and transfer, differentiable thickness and morphing, neural shape reconstruction, and impact localization with acoustic transfer. Audio and neural networks execute through PyTorch MPS on Metal in float32. FEM assembly, eigenvalue sensitivities, and SciPy's sparse eigensolver use CPU float64. The released boundary-element method runs on CPU through Bempp's Numba backend.

**Implementation coverage is not a reproduction of all published figures.** The five released inference workflows retain their source settings. Impact localization, material transfer and the loss ablation are paper-described reconstructions because their experiment drivers are absent. The release also lacks their complete mesh selection, microphone calibration and localization ground truth. Those choices are identified below and in run metadata.

## Install and obtain the inputs

From the AcousticSolver root on Apple Silicon with Python 3.12:

```sh
git submodule update --init external/diffsound
python3.12 -m venv build/reference/diffsound-env
build/reference/diffsound-env/bin/pip install -r script/diffsound/requirements.txt
build/reference/diffsound-env/bin/python -m script.diffsound.fetch
```

The fetch command pins revision `3a0be14b8bfa95dcbcd98f3c233e981c713d021b`, archives the source and released assets under `data/diffsound/source`, and records SHA-256 hashes. Subsequent calls verify the archive. Training uses this local archive, without the author checkout, CUDA or a remote machine. The released data includes the ceramic bowl mesh and eight microphone recordings, shape meshes, and tetrahedral grids. See [NOTICE](NOTICE) for source attribution.

## Run

```sh
build/reference/diffsound-env/bin/python -m script.diffsound.run material-real \
  --output data/diffsound/runs/bowl
```

The task selects an experiment, with one implementation of each component:

| Task | Inputs and optimization |
| --- | --- |
| `material-synthetic` | Sixteen bowl fits with independently sampled initial/target materials; quadratic targets; 16 modes; 3,000 updates |
| `material-real` | Eight released bowl recordings; 512 free modes and 2,001 damping updates; then 16 FEM modes and 3,000 material updates |
| `thickness` | Frog shell targets at 0.3–0.7; 128-grid generation, 64-grid fitting; 32 modes; 500 updates per target |
| `morphing` | Frog/turtle interpolations at 0–1; 128-grid generation, 64-grid fitting; 16 modes; 10 updates per target |
| `geometry` | Bob, oloid and spot; 16³ occupancy samples, 32 tetrahedral grid; up to 2,000 SDF pretraining updates and 1,000 shape updates using 64 target eigenvalues |

Defaults live in [run.py](run.py). Material fitting uses transport loss for 1,000 updates, then resets Adam and uses spectral loss for 2,000 updates. Eigenpairs refresh every 15 updates. `exp_mode` selects the released experiment conditions: 0/1 fit Young's modulus with linear/quadratic FEM, and 2/3 also fit Poisson's ratio. Density stays fixed; sound identifies the Young's-modulus/density ratio rather than both independently.

Use `--config overrides.json` to override named settings and `--seed N` for reproducible local initialization. Every effective setting is recorded. Changed settings are an adaptation of the released schedule. For example, `{"steps":30,"early_steps":15}` provides a bounded material run. Output directories must be new. Resume with the same command plus `--resume`; configuration, input and implementation hashes must match.

SIGINT or SIGTERM saves after the current optimizer update. Checkpoints contain model, optimizer, scheduler, eigenpair cache, CPU/MPS random states and progress. Results include parameters, arrays and artifact hashes. Material runs export target/predicted WAVs; shape runs export meshes. Keep wanted runs under `data/diffsound/runs`; development checks belong under `build/`.

## Impact localization and loss ablation

```sh
build/reference/diffsound-env/bin/python -m script.diffsound.impact \
  --material-run data/diffsound/runs/bowl \
  --listener 0.5 0.4 0.6 --output data/diffsound/runs/bowl-impact

build/reference/diffsound-env/bin/python -m script.diffsound.ablation \
  --output data/diffsound/runs/loss-ablation
```

The listener coordinates above are an example, **not the released microphone calibration**. Supply the microphone position in the mesh's coordinate system. Localization jointly refines material and per-recording modal gains, starting from a material run, then evaluates unit normal impulses at surface vertices. Acoustic transfer uses the released exterior Helmholtz boundary equation. Cosine similarity between fitted and simulated modal gains gives a surface score and maximum-score vertex. It is not a calibrated probability, and symmetric objects can have indistinguishable strike positions. Air density 1.225 kg/m³, sound speed 343 m/s, vertex-normal forces and the refinement schedule are explicit reconstruction choices. The acoustic solve is a post-fit localization step, outside the differentiable optimizer.

The ablation optimizes a scalar eigenvalue multiplier with spectral, transport, or hybrid loss for ceramic, iron, steel and glass. It uses the released bowl mesh for all four materials; it does not claim to reproduce Figure 4's unavailable four-object selection. Both drivers support `--steps`, `--early-steps` and `--resume`.

For Figure 8's material-transfer procedure, supply a destination tetrahedral mesh and onset-aligned recordings:

```sh
build/reference/diffsound-env/bin/python -m script.diffsound.transfer \
  --material-run data/diffsound/runs/bowl --mesh plate.msh --audio plate.wav \
  --output data/diffsound/runs/plate
```

This fixes the source material and damping curve, recomputes the destination modes and fits modal gains and filtered noise. Audio is resampled, converted to mono, peak-normalized and truncated to the source run's duration. The mesh must use the same length units. The paper's plate mesh/recordings and transfer driver are absent; the 3,000-update Adam schedule and noise gain use explicit local settings. `--steps` and `--resume` are supported.

## Fidelity and scope

- Source settings take precedence where the paper and release differ: fixed loss switch instead of a plateau detector, 32 shell/morph coefficient weights instead of 16, and the released 16-mode/10-update morphing schedule. Damping selection uses minimum damping below 100 in each frequency bin, as the code does, rather than the paper's amplitude-pruning description.
- CPU regeneration of synthetic targets is an adaptation of the CUDA release. CPU float64 FEM accumulation and deterministic ARPACK initialization can change numerical trajectories. MPS uses float32 audio/neural arithmetic; phase integration uses constant-frequency multiplication equivalent to the source's cumulative sum, with different rounding.
- The transport objective preserves detached spectral features and the source's repeated-index coordinate gradients. Its recorded-audio and synthetic paths have different frequency-coordinate multiplicities. GeomLoss uses its tensorized Sinkhorn backend instead of requiring CUDA KeOps.
- Shape tasks use target eigenvalues and, for geometry, coarse occupancy observations. Each experiment supplies a tetrahedral mesh, a shape family or coarse occupancy constraints. Neural SDF clipping can change mesh topology; it does not learn scene/contact graph topology from raw recordings.
- Material fitting assumes the released known impulse. Localization estimates modal gains and strike position, not an unknown contact-force trajectory. Joint contact/body estimation remains available in the DiffImpact port; this port does not substitute a known impulse for that broader research goal.
- No image/video encoder, nonlinear-contact model, Faust export, or real-time FEM solver is claimed by this port. These are outside the released DiffSound methods.

## Validate

```sh
script/ValidateDiffSound
```

See [validation coverage](../../VALIDATION.md#diffsound) for fixture provenance and paper-reproduction limits. `python -m script.diffsound.capture` recaptures the pinned author comparison when deliberately auditing source parity.
