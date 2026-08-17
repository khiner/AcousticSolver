# AcousticSolver
Exploring acoustic radiation transfer GPU solvers

## WaveBlender

A from-scratch Metal port of [WaveBlender](https://github.com/kangruix/WaveBlender)
(Xue et al., SIGGRAPH Asia 2024), an FDTD acoustic wave solver that renders sound for
animated scenes from coupled sources: vibrating rigid bodies (modal), water bubbles,
speakers, point impulses, and occluders. Output is validated per-scene against golden
outputs from the CUDA reference — see [VALIDATION.md](VALIDATION.md).

Wall-clock per scene, full `script/ValidateGolden` runs. CUDA reference on an RTX 4090
(Linux, CUDA 12.8), Metal on an Apple M5 Max:

| Scene | RTX 4090 | Initial Metal port | Current |
|---|---|---|---|
| CupPhone | 21.5s | 56.2s | 34.7s |
| GlassPour | 41.4s | 34.9s | 5.3s |
| LegoDrop | 13.7s | 10.0s | 3.4s |
| SpollingBowl | 83.5s | 61.3s | 9.5s |
| FillerUp | 72.2s | 90.2s | 30.1s |
| Trumpet | 45.4s | 76.8s | 38.3s |
| TalkFan | 132.2s | 128.2s | 54.0s |
| HandShake | 102.5s | 124.9s | 54.1s |
| 2016Pour | 127.4s | 187.0s | 79.8s |
| PaddleSplash | 615.1s | 400.5s | 60.7s |
| **Total** | **1254.9s** | **1170.0s** | **369.9s** |

Notable performance changes relative to upstream WaveBlender:

- One fused full-grid kernel per FDTD step (velocity + pressure, ping-pong buffers)
  instead of separate passes, with byte cell states and blending weights derived
  in-kernel.
- The GPU runs batch N while the CPU prepares batch N+1, with one sync per batch. The
  next batch's command buffers are encoded and committed before the sync (gated on a
  shared event the host signals after its fresh-cell writes), and the fresh-cell
  least-squares systems are factorized pre-sync, so the GPU restarts almost immediately.
- Bubble mass-matrix inverses are precomputed on worker threads with Accelerate (AMX)
  and chained across adjacent event intervals by low-rank updates, so the coupled-bubble
  solve applies two matrix-vector products per RK4 stage.
- The fresh-cell least-squares solve runs per connected component, in parallel.
- Modal filter coefficients are computed once, and modal transfer-matrix rows are reused
  across batches when the pose and boundary face are unchanged.
- Flat, data-oriented CPU batch prep: lookup tables instead of sets/maps, bounds-limited
  grid sweeps, time-windowed impulse queries, and parallel rasterization and
  closest-point queries.
