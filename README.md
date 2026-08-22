# AcousticSolver
Exploring acoustic radiation transfer GPU solvers

## WaveBlender

A from-scratch Metal port of [WaveBlender](https://github.com/kangruix/WaveBlender) (Xue et al., SIGGRAPH Asia 2024), an FDTD acoustic wave solver that renders sound for animated scenes from coupled sources: vibrating rigid bodies (modal), water bubbles, speakers, point impulses, and occluders.
Output is validated per-scene against the CUDA reference's listener outputs, committed under `gen/cuda/` — see [VALIDATION.md](VALIDATION.md), which also covers the five places this deliberately departs from the reference, including a modal timestep bug that puts the three Modal scenes at envelope rather than waveform agreement until the references are regenerated.

The ~12 GB of scene data is not vendored: run `script/FetchScenes` to download it into `Scenes/` (checksummed against `script/scenes.sha256`).
Ten scenes come from the WaveBlender dataset and include their own `config.json`. Cymbal and WineglassTap come from the earlier wavesolver dataset ([Wang et al. 2018](https://graphics.stanford.edu/projects/wavesolver/dataset/dataset_table.html)), which WaveBlender used but never re-published.
The fetch script renames and moves their files into our layout, and their configs are in `config/`.

Wall-clock per scene. CUDA reference on an RTX 4090 (Linux, CUDA 12.8), Metal on an Apple M5 Max.
The Metal column is a cool sweep — one scene per idle GPU, two minutes apart, on an otherwise
quiet machine — because this GPU's thermal drift is large enough (a scene measures 43s cool and
53s heat-soaked) that back-to-back suite timings are not comparable across runs:

| Scene | RTX 4090 | Initial Metal port | Current |
|---|---|---|---|
| CupPhone | 21.5s | 56.2s | 28.9s |
| GlassPour | 41.4s | 34.9s | 5.3s |
| LegoDrop | 13.7s | 10.0s | 2.7s |
| SpollingBowl | 83.5s | 61.3s | 6.3s |
| FillerUp | 72.2s | 90.2s | 26.9s |
| Trumpet | 45.4s | 76.8s | 24.9s |
| TalkFan | 132.2s | 128.2s | 33.3s |
| HandShake | 102.5s | 124.9s | 45.6s |
| 2016Pour | 127.4s | 187.0s | 51.6s |
| PaddleSplash | 615.1s | 400.5s | 51.6s |
| Cymbal | 67.4s | — | 10.2s |
| WineglassTap | 32.0s | — | 2.7s |
| **Total** | **1354.3s** | **1170.0s** (ten scenes) | **290.0s** |

Cymbal and WineglassTap were added after the initial port, so the middle column is empty for
them. Cymbal's 10.2s reads its 11 GB of per-frame shell data from the page cache. The first read
from disk takes 40.9s. The reference read the same data from a RAM disk.

Notable performance changes relative to upstream WaveBlender:

- Three scenes step at the grid's Courant limit rather than at the audio rate they inherited (TalkFan/Trumpet 88.2 → 66.15 kHz, 2016Pour 192 → 128 kHz): 1.34–1.5× fewer timesteps, and *less* numerical dispersion, since this scheme's spatial and temporal truncation errors cancel most fully at the stability limit. Retimed configs live in `config/`; the CUDA reference was regenerated at the matching rates so the validation verdicts are unchanged.
- One fused full-grid kernel per FDTD step (velocity + pressure, ping-pong buffers) instead of separate passes, with byte cell states and blending weights derived in-kernel.
- PML split-pressure fields packed into dense shell slabs (full-grid storage left most of each cache line untouched in the x-tapered strips), and threadgroup shapes chosen per scene grid size.
- Boundary conditions applied to the velocities as they are written by the fused step, rather than by a separate dispatch per timestep, so each thread transforms exactly the faces it owns.
  A runtime probe picks this or the standalone dispatches per scene.
- The GPU runs batch N while the CPU prepares batch N+1, with one sync per batch.
  The next batch's command buffers are encoded and committed before the sync (gated on a shared event the host signals after its fresh-cell writes), and the fresh-cell least-squares systems are factorized pre-sync, so the GPU restarts almost immediately.
  Batches with no fresh cells — half of them in a typical animated scene, all of them in some — have nothing for the host to write, so their gate opens *before* the sync and the host round trip leaves the GPU's critical path entirely.
- Bubble mass-matrix inverses are precomputed on worker threads with Accelerate (AMX) and chained across adjacent event intervals by low-rank updates, so the coupled-bubble solve applies two matrix-vector products per RK4 stage.
  Those two products are independent, so they run on separate cores once an inverse outgrows the cache, which also keeps the solve thread out of the precompute workers' memory bandwidth.
- The fresh-cell least-squares solve runs per connected component, in parallel.
- Modal filter coefficients are computed once, and modal transfer-matrix rows are reused across batches when the pose and boundary face are unchanged.
- Flat, data-oriented CPU batch prep: lookup tables instead of sets/maps, bounds-limited grid sweeps, time-windowed impulse queries, and parallel rasterization and closest-point queries.
