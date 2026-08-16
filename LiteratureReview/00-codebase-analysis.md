# WaveBlender Codebase Analysis (port target)

Source: `../WaveBlender` — reference implementation of *WaveBlender: Practical Sound-Source Animation in Blended Domains* (Xue, Wang, Langlois, James, SIGGRAPH Asia 2024). C++17 + CUDA + Eigen 3.4. ~1,000 lines of CUDA across the core solver and shaders, plus two CPU sub-libraries.

## Architecture

The simulation runs in **batches** at `blendrate` Hz (default 100 Hz, i.e. 10 ms batches). Each batch:

1. **Rasterize** animated triangle meshes into the grid (`WaveBlender.cu`, CPU-side, uses `tribox.h` triangle–box overlap and AABB trees), producing per-cell object IDs (`d_cell`) at batch endpoints t1/t2.
2. **Cavity detection** (flood fill, CPU) marks enclosed air pockets (`CAVITY_INTERIOR = 255`).
3. **Fresh-cell extrapolation** initializes pressure/velocity in cells that switched from solid to air.
4. **Shader evaluation**: each `Object`'s acoustic shader computes boundary-velocity (or force) time series for the batch at `shader_srate` (48/44.1 kHz) into a flat device buffer `d_shaderData`, indexed by `d_shaderMap` (sign encodes velocity-BC vs. force, value encodes cell + axis).
5. **FDTD timestepping** (`GPUSolver.cu`): `FDTD_srate / blendrate` steps (e.g. 882–1200 per batch), three kernels per step.

## The three core kernels (per timestep)

- `Ker_stepPressure` — staggered-grid divergence update. Split pressure field (`d_px/py/pz`) inside an 8-cell PML shell (Berenger split-field, weights precomputed as 1D lookup tables `pmlN`/`pmlD` indexed by distance-to-boundary). Also advances the blending field `beta` with a cubic smoothstep in normalized batch time.
- `Ker_applyShader` — scatters interpolated shader samples into velocity faces: `v += beta*(vb − v)` for velocity BCs, `v += (1−beta)*val*dt/rho` for forces. Linear interpolation between shader samples (48 kHz → 88.2/120 kHz).
- `Ker_stepVelocity` — pressure-gradient update, blended by `max(beta, beta_neighbor)` per face, with velocity PML weights.

Launch config: 8×8×8 thread blocks over the full grid (~88³–100³ cells typical); shader kernel is 1D over boundary points. Listeners (`GPUListener.h`) accumulate `d_p` at a cell index each step into a device buffer, written to a `.bin` file per batch (converted to WAV by `scripts/write_wav.py`).

## Acoustic shaders (Source/Shaders/)

| Shader | Sound model | GPU usage |
|---|---|---|
| `Monopole` | analytic monopole source | trivial kernel |
| `Speaker` | plays a WAV through mesh faces (e.g. phone speaker) | custom kernel + cuBLAS handle |
| `Occluder` | rigid non-vibrating boundary | rasterization only |
| `Modal` | modal vibration + acceleration noise of rigid bodies (`ModalSound` lib) | mode-to-boundary projection (custom kernel, `// TODO: port to cublas`); modal ODEs on CPU |
| `Bubbles` | coupled-bubble water sound (`FluidSound` submodule) | `cublasSgemv` flux normalization; bubble ODEs on CPU in `double` |
| `Shell` | thin-shell vibration from per-vertex displacement animations | custom kernel |
| `Point` | point source array | custom kernel |

`ModalSound` is based on Jui-Hsien Wang's **openpbso** (github.com/jhwang7628/openpbso); it reads precomputed modal data (`.modes` eigen-decompositions, material params) and impact records. `FluidSound` (github.com/kangruix/FluidSound, git submodule, **not yet checked out locally**) implements the coupled-bubble oscillator solver from Xue et al. 2023.

## CUDA API surface to replace

- `cudaMalloc`/`cudaFree`/`cudaMemcpy` (H2D per batch for shader data + rasterized cell states; D2H per batch for listener output and debug slices) → `MTLBuffer` (Apple-silicon unified memory makes most copies unnecessary).
- Kernel launches: 3 kernels × ~1000 steps per batch, all on the default stream (implicit ordering) → one `MTLCommandBuffer` per batch encoding all steps, or split for listener readback.
- `cublasSgemv` (bubble flux), `cublasCreate` handles in Speaker/Modal/Bubbles → MPS `MPSMatrixVectorMultiplication` or a small custom kernel (the matrices are modest: N_points × N_modes/N_bubbles).
- Notably absent: no atomics, no `__shared__` memory, no `__syncthreads`, no streams or events anywhere in the CUDA code (one `cudaDeviceSynchronize` in `BubbleShader.cu`). Every kernel is an embarrassingly parallel gather/scatter — the Metal translation is nearly mechanical.

## Precision notes

- `REAL = float` throughout the grid solver. All device math is fp32 → **no fp64 blocker on the GPU side**.
- `double` is used on the CPU: FluidSound bubble ODE solver (`FluidSound::Solver<double>`), ModalSound IIR modal integrator, timestamps/blending times. These stay CPU-side in the port, unchanged.
- The `cublasDgemv` call in `BubbleShader.cu` sits behind the `REAL`-type branch and is dead when `REAL = float`.

## Inputs / outputs (for validation)

- Scene config: JSON (`Scenes/*/config.json`) — grid dims, `cellsize`, `FDTD_srate`, `shader_srate`, `blendrate`, object list (mesh OBJ, WAV, animation txt, shader type), listener positions.
- Example scene `CupPhone` included (phone playing marimba.wav inside a cup, 88³ grid @ 88.2 kHz, 8 s). More scenes in the paper's dataset (see 04-data-and-validation.md).
- Output: raw float32 pressure `.bin` → `scripts/write_wav.py` → WAV. Also `logZSlice` debug dumps of pressure+beta slices — useful as a golden-image comparison channel between CUDA and Metal builds.

## Port-relevant observations

- The solver is memory-bandwidth-bound stencil code with no shared-memory tiling ("some optimizations have been removed in favor of readability") — a straightforward kernel-for-kernel port is viable, with optimization later.
- Ordering constraint from the README: point sources must be rasterized last; rasterization order matters.
- Per-step listener sampling reads one cell of `d_p` per listener — on Metal, do this in-kernel (append to a device buffer) rather than syncing per step.
- ~100k timesteps/simulated-second × 3 dispatches each ⇒ dispatch overhead is the main new performance risk on Metal; batching an entire batch's steps into one command buffer is the standard remedy.
