# AcousticSolver
Metal GPU acoustic wave solvers: sound sources in animated scenes (WaveBlender), exterior radiation without ghost cells (SonicRadiation), and room impulse responses with impedance walls (Bilbao/Hamilton FDTD).
Each is validated against its reference implementation or an analytic ladder — see [VALIDATION.md](VALIDATION.md).

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

## SonicRadiation

A second, independent solver implementing [SonicRadiation](https://arxiv.org/abs/2508.08775) (Jin, Zhu, Wang, Li — CAVW/CASA 2026), written from the paper; no reference implementation is public.
It shares this repo's Metal runtime, scene configs, mesh loading and listener output with the WaveBlender path, and nothing else — `src/Radiation/` is a parallel solver, not a mode of the first one.

Where WaveBlender rasterizes solids into the grid and reconciles the boundary with ghost cells, fresh-cell extrapolation and per-cell blending weights, this method deletes that entire stack. Every cell is an air cell. A time-domain BEM (BDF2 convolution quadrature) carries the near-field boundary potentials on the object's own triangles, a scalar second-order FDTD grid carries far-field transport, and the two couple through purely local per-step operations:

- Each cell stores only its **far-field** pressure — the contribution of boundary elements outside its Chebyshev-R1 neighbourhood. Neighbour terms in the Laplacian are corrected for the set difference between adjacent cells' far sets by direct retarded-potential evaluation (the paper's Eq. 12).
- Element Dirichlet values close the system each step: near elements by direct BEM sums, far elements by trilinear interpolation of grid far-field values re-based onto the element's own far set (Eqs. 17–20).
- The domain is terminated by the same quadratic-ramp split-field PML as the main solver, run as the first-order pressure–velocity system in an 8-cell shell. Eliminating the velocities from that staggered scheme yields exactly the interior's scalar update, so the seam adds no scheme mismatch beyond the PML's own reflection floor.

There is no per-batch host round trip in the time loop at all — no fresh cells to solve, no rasterization to sync on. A batch uploads its Neumann samples and runs.

```
build/AcousticSolver --radiation config/WineglassTap.json                # render a scene
build/AcousticSolver --radiation --seconds 0.3 config/WineglassTap.json  # just the first 0.3s
build/RadiationTest                                                     # the validation ladder (~0.7s)
build/RadiationTest --mesh <obj> --cell <metres>                        # what remeshing does to a scene mesh
build/RadiationTest --stability --time 10                               # long-run growth probe
build/RadiationTest --move 2.0 --epoch 16                               # dynamic geometry
script/ValidateRadiation                                                # the regression gate (~8s)
script/RadiationBench --rounds 3 RadiationTest --v2 --res 60            # interleaved A/B against another commit
script/RadiationModes                                                   # per-mode cross-check against the WaveBlender path
```

The gate is the ladder under `RadiationTest --gate`, where every rung is checked against the value recorded in [VALIDATION.md](VALIDATION.md) rather than merely printed, plus a scene render byte-compared against `gen/radiation/`. `config/WineglassTap_move.json` is the moving scene it renders — 50 ms of the wineglass translating and rotating, 11 geometry epochs — since every config from the dataset carries an identity animation track and would leave the epoch path uncovered.

A body that moves follows its scene's animation track: the geometry tables are rebuilt at epochs paced by how far the body has travelled rather than by a step count, each set built for the midpoint of the interval it serves, on a worker thread while the GPU steps the previous one. Two things make a rebuild affordable enough to hide. The convolution-quadrature weights of every non-self pair are a scaled sum of two universal functions of the retarded delay, tabulated once, so a rebuild costs no contour transforms at all; and an epoch carries its element-element weights over from the last one, since rigid motion leaves every quantity they are built from alone. WineglassTap's scene build is 1.8s against 6.3s before, of which the tables are 0.78s. An epoch after the first also builds its weights straight into the half-precision layout the GPU reads, so the float copies — some 2.5 GB for this scene — are never allocated.

Scope is what the method covers: closed, rigid, modal bodies. Water, speakers, point impulses, occluders and thin shells stay on the WaveBlender path. Listener output is written on both paths the method offers — the grid sample and the Eq. 5 retarded-potential evaluation — into `build/radiation/` as `<output>_grid.bin` and `<output>_eq5.bin`, with the step rate in `<output>.json` (it runs at the grid's Courant limit, not the config's FDTD rate). They have opposite error profiles (grid dispersion versus convolution-quadrature frequency warping), so which one wins depends on the listener's distance.

WineglassTap renders in under eight minutes on an M5 Max — 118,888 steps at the grid's Courant limit, over 13,852 boundary elements and 1.2 GB of convolution-quadrature tables. About four fifths of a step is the two convolution passes over those tables; the scalar FDTD interior that the WaveBlender path spends nearly all its time in is 0.4% here. Those passes are bound by the weight stream rather than by the arithmetic over it, which is why the weights are signed bytes with a power-of-two scale per lag window. The method buys the vanished host round trip and a boundary treatment with no ghost cells, and pays for it in the boundary integral.

Departures from the paper, all forced by measurement (see [VALIDATION.md](VALIDATION.md)):

- The paper's literal near-set radii leave the boundary–grid feedback loop with gain slightly above one. The defaults here are R1=2 (rather than 1) plus a one-zero filter on the interpolated far-field feedback, which nulls the grid-Nyquist mode the loop amplifies.
- Element counts, and therefore every table built over them, scale near-quadratically with mesh density, so scene meshes are retargeted to the grid by incremental isotropic remeshing before anything else happens.

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
