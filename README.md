# AcousticSolver
Metal GPU acoustic wave solvers: sound sources in animated scenes (WaveBlender), exterior radiation without ghost cells (SonicRadiation), and room impulse responses with impedance walls (Bilbao/Hamilton FDTD).
Each is validated against its reference implementation or an analytic ladder — see [VALIDATION.md](VALIDATION.md).

## WaveBlender

A from-scratch Metal port of [WaveBlender](https://github.com/kangruix/WaveBlender) (Xue et al., SIGGRAPH Asia 2024), an FDTD acoustic solver for animated scenes with coupled modal bodies, water bubbles, speakers, point impulses, and occluders.
Per-scene listener output is validated against the CUDA reference under `gen/cuda/`.
[VALIDATION.md](VALIDATION.md) also documents five intentional departures, including a reference modal-timestep bug that limits the three Modal scenes to envelope agreement until their goldens are regenerated.

The approximately 12 GB scene dataset is not vendored.
`script/FetchScenes` downloads it into `Scenes/` and verifies it against `script/scenes.sha256`.
Ten scenes come from the WaveBlender dataset with their own `config.json`; Cymbal and WineglassTap come from the earlier wavesolver dataset ([Wang et al. 2018](https://graphics.stanford.edu/projects/wavesolver/dataset/dataset_table.html)), which WaveBlender used but did not republish.
The fetch script maps those two scenes into this repository's layout, with configs under `config/`.

Wall-clock per scene. CUDA reference on an RTX 4090 (Linux, CUDA 12.8), Metal on an Apple M5 Max.
The Metal column is a cool sweep: one scene per idle GPU, two minutes apart, on an otherwise quiet machine.
Thermal drift can move one scene from 43s cool to 53s heat-soaked, so back-to-back suite timings are not comparable across runs.

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

Cymbal and WineglassTap were added after the initial port, so they have no middle-column result.
Cymbal's 10.2s result reads its 11 GB of per-frame shell data from the page cache; a cold read takes 40.9s, while the reference used a RAM disk.

Notable performance changes relative to upstream WaveBlender:

- Three scenes use the grid's Courant limit instead of their inherited audio rate (TalkFan/Trumpet 88.2 → 66.15 kHz, 2016Pour 192 → 128 kHz). This takes 1.34–1.5× fewer timesteps and reduces numerical dispersion because the spatial and temporal truncation errors cancel most fully at the stability limit. Retimed configs live in `config/`, with regenerated CUDA references at matching rates.
- One fused full-grid kernel per FDTD step (velocity + pressure, ping-pong buffers) instead of separate passes, with byte cell states and blending weights derived in-kernel.
- PML split-pressure fields packed into dense shell slabs instead of sparse full-grid storage, with threadgroup shapes selected for each scene's grid size.
- Boundary conditions applied as the fused step writes velocity, so each thread transforms only the faces it owns. A runtime probe chooses between this path and standalone dispatches for each scene.
- The GPU runs batch N while the CPU prepares batch N+1, with one synchronization per batch. Command buffers are committed before the sync behind a shared-event gate, and fresh-cell systems are factorized in advance. Batches without fresh cells open the gate before the sync and remove the host round trip from the GPU's critical path.
- Bubble mass-matrix inverses are precomputed on Accelerate worker threads and chained across adjacent event intervals with low-rank updates. The coupled-bubble solve then applies two independent matrix-vector products per RK4 stage, using separate cores once an inverse outgrows the cache.
- The fresh-cell least-squares solve runs per connected component, in parallel.
- Modal filter coefficients are computed once, and modal transfer-matrix rows are reused across batches when the pose and boundary face are unchanged.
- Flat, data-oriented CPU batch prep: lookup tables instead of sets/maps, bounds-limited grid sweeps, time-windowed impulse queries, and parallel rasterization and closest-point queries.

## SonicRadiation

`src/Radiation/` is an independent implementation of [SonicRadiation](https://arxiv.org/abs/2508.08775) (Jin, Zhu, Wang, Li — CAVW/CASA 2026), written from the paper because no reference implementation is public.
It shares the Metal runtime, scene format, mesh loading, and listener output with WaveBlender but uses a separate solver.

SonicRadiation couples a time-domain boundary-element method on the object's triangle mesh to a scalar FDTD grid for far-field transport.
The GPU time loop requires no fresh-cell solve, rasterization synchronization, or other per-batch host round trip.
Moving rigid bodies are supported by rebuilding geometry-dependent data between simulation epochs.

The method targets closed, rigid, modal bodies.
Water, speakers, point impulses, occluders, and thin shells remain on the WaveBlender path.
Listener output is available both as a grid sample and through the paper's retarded-potential evaluation, written under `build/radiation/` as `<output>_grid.bin` and `<output>_eq5.bin` with metadata in `<output>.json`.

```
build/AcousticSolver --radiation config/WineglassTap.json
build/AcousticSolver --radiation --seconds 0.3 config/WineglassTap.json
build/RadiationTest
script/ValidateRadiation
script/RadiationModes
```

`script/ValidateRadiation` checks analytic behavior, stability, moving geometry, and deterministic scene output against the committed golden.
`script/RadiationModes` cross-checks modal output against WaveBlender.
[VALIDATION.md](VALIDATION.md) records the numerical methods, implementation departures, thresholds, and detailed results.

## Room acoustics

`src/Room/` renders room impulse responses with the finite-volume FDTD boundary treatment of [Bilbao, Hamilton, Botts & Savioja 2016](https://www.pure.ed.ac.uk/ws/files/22154168/fv_genimp_final_r3.pdf).
Scenes choose either a 7-point Cartesian grid or a 13-point face-centred-cubic grid and may use rigid or parallel-LRC material boundaries.

The explicit solver consumes converted [PFFDTD](https://github.com/bsxfun/pffdtd) voxelizer output so the Metal and reference engines step the same discrete problem.
`script/ConvertRoomScene` writes the scene configuration and binary boundary, material, source, and receiver data under `Scenes/`.
The repository includes Cartesian and FCC shoebox and church scenes, a Cartesian concert hall, and a production-resolution FCC shoebox.

The optimized 27-point implicit scheme from [Smits & Bilbao 2025](https://doi.org/10.1121/10.0036229) runs directly from a PFFDTD `model_export.json`.
It voxelizes all 26 stencil legs, preserves mesh normals and materials, and applies the paper's staircase-compensated real-admittance boundary.
Its defaults reproduce the published Fig. 6 numerical configuration; zero-mean projection for long fp32 records is available separately and remains off by default.

```
build/AcousticSolver --room ../Scenes/RoomChurch/config.json
build/AcousticSolver --room --seconds 0.2 ../Scenes/RoomChurch/config.json
build/AcousticSolver --implicit-room path/to/model_export.json
build/AcousticSolver --implicit-room --h 0.02 --seconds 0.1 path/to/model_export.json
build/RoomTest --implicit
script/ValidateRoom
script/ConvertRoomScene RoomChurch
script/RunRoomReference RoomChurch
script/RenderRoomWavs
```

`script/ValidateRoom` runs the analytic, energy, stability, CUDA-golden, and Metal-determinism gates for the explicit schemes.
`RoomTest --implicit` covers dispersion, boundaries, voxelization, stability, and projection for the implicit scheme.
[VALIDATION.md](VALIDATION.md) records the methods, thresholds, and detailed results.

Raw differentiated-pressure receiver rows are written to `build/room/<output>.bin`, with the sample rate in the adjacent JSON file.
`script/RenderRoomWavs` applies the reference post-processing and writes normalized 48 kHz WAVs under `gen/wav/room/`.
