# Literature Review: WaveBlender CUDA → Metal Port

Compiled 2026-08-15. Everything needed to port `../WaveBlender` (FDTD acoustic wavesolver, SIGGRAPH Asia 2024) to Metal on Apple Silicon, completely and with validation.

- **[00-codebase-analysis.md](00-codebase-analysis.md)** — what the CUDA code actually does: batch pipeline, the three core kernels, acoustic shaders, CUDA API surface (all fp32 on-device, no atomics/shared memory/streams), inputs/outputs.
- **[01-primary-papers.md](01-primary-papers.md)** — the WaveBlender paper (PDF, video, equations to reproduce) and its full lineage: Wang 2018 (acoustic shaders, fresh-cell pressure), Aerophones 2015 (β-blending ancestor), modal/bubble/shell/point-source model papers, repo status.
- **[02-numerical-methods.md](02-numerical-methods.md)** — FDTD fundamentals (Yee, Botteldooren, Taflove & Hagness, Bilbao), split-field PML theory and expected reflection levels, ghost/fresh-cell immersed-boundary literature, GPU FDTD implementations (ParallelFDTD, PFFDTD), analytic validation solutions.
- **[03-metal-porting.md](03-metal-porting.md)** — Apple docs and WWDC sessions, CUDA→Metal concept map, metal-cpp + CMake, MPS for the cuBLAS calls, precision/fast-math traps, exemplar repos (MLX, llama.cpp, BabelViscoFDTD, Gebraad & Fichtner's M1 FDTD paper), dispatch-batching strategy for ~100k timesteps/s.
- **[04-data-and-validation.md](04-data-and-validation.md)** — the 10-scene WaveBlender dataset (formats, sizes), and a three-tier validation plan: analytic unit tests → CUDA-vs-Metal tolerance comparison → end-to-end scene/audio checks.

## Headline conclusions

1. **The port is tractable and nearly mechanical at the kernel level.** The GPU code is bandwidth-bound fp32 stencil kernels with no shared memory, atomics, or streams. Doubles exist only in CPU-side ODE solvers (unaffected by Metal's lack of fp64).
2. **Closest precedents**: Gebraad & Fichtner 2023 (C++ seismic FDTD on M1 via MSL, ~10× over CPU) and BabelViscoFDTD (multi-backend staggered-grid FDTD where M1 Max beats discrete GPUs). **No Metal FDTD *acoustics* codebase exists — this port is first of its kind.**
3. **Two real risks**: (a) dispatch overhead at 3 dispatches × ~100k steps/s — solved by encoding whole batches into single command buffers with 2–3 in flight; (b) Metal's default fast-math — must be `safe` for validation builds, and bit-for-bit CUDA equality is impossible, so validation is tolerance-based against analytic solutions and CUDA/CPU references.
4. **Data is all available**: 10 scenes (~600 MB) on the Stanford dataset page, plus the in-repo CupPhone scene. The FluidSound submodule must be fetched (`git submodule update --init --recursive`) and should be vendored — its GitHub repo was archived in May 2026.
5. **Upstream is stable**: WaveBlender is a finished research artifact; the authors' promised next-gen system has not appeared.
