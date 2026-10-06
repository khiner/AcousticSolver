# AcousticSolver

Offline acoustic solvers for Apple Silicon, implemented in C++ and Metal.

| Solver | Use | Scope |
| --- | --- | --- |
| [WaveBlender](https://github.com/kangruix/WaveBlender) | Sound from animated scenes | Modal bodies, bubbles, speakers, impulses and occluders |
| [SonicRadiation](https://arxiv.org/abs/2508.08775) | Exterior sound radiation | Closed rigid modal bodies, including motion |
| [Room acoustics](https://www.pure.ed.ac.uk/ws/files/22154168/fv_genimp_final_r3.pdf) | Room impulse responses | Cartesian/FCC FDTD with rigid or frequency-dependent walls |
| [Implicit room acoustics](https://doi.org/10.1121/10.0036229) | Low-dispersion room simulation | 27-point scheme with real-admittance walls |
| [Immersed boundaries](https://doi.org/10.1121/10.0020635) | Exterior scattering and transmission | Static surfaces, impedances and finite barriers |
| [Curved tetrahedral DG](src/Dg/README.md) | Rigid-wall wave propagation | Prepared meshes and operators; no native scene/source integration |
| [Nonlinear plates](https://arxiv.org/abs/2608.06139) | Plate vibration and crash sounds | Simply supported rectangular plates; displacement output |
| [Nonlinear stiff strings](https://doi.org/10.1007/s11071-026-12708-0) | Plucked-string vibration | Geometrically exact, cubic and Kirchhoff–Carrier models; displacement output |
| [Acoustic Reliefs](script/reliefs/README.md) | Acoustic heightfield optimization | Differentiable boundary elements with an appearance objective |

See [validation](VALIDATION.md) for checks and numerical limits, and [third-party notices](NOTICE.md) for licensing and reference-code provenance.

## Build

Requires macOS on Apple Silicon, Xcode's Metal toolchain, CMake 3.24+, and a C++23 compiler. Homebrew LLVM is used for development. Boost, MPFR and GMP support plate preparation; Eigen is vendored and CMake downloads pinned metal-cpp headers.

```sh
brew install cmake ninja llvm boost mpfr gmp
PATH="$(brew --prefix llvm)/bin:$PATH" script/Build
```

`script/Build --debug` selects a debug build. Python scoring and WAV tools require NumPy, SciPy and SoundFile. Plate and other reference wrappers manage their own isolated Python environments; [Acoustic Reliefs](script/reliefs/README.md) has separate dependencies.

Commands below start at the repository root unless indicated otherwise. Run the WaveBlender, SonicRadiation and room examples from `build/` as shown; their dataset paths or output directories are relative to the working directory.

## WaveBlender

The scene dataset is about 12 GB and is downloaded separately. Scene-specific overrides live in `config/`; downloaded files under `Scenes/` are checksummed.

```sh
script/FetchScenes CupPhone WineglassTap   # omit names to fetch every scene
script/FetchScenes --verify
(cd build && ./AcousticSolver ../Scenes/CupPhone/config.json)
script/ValidateGolden CupPhone
script/RenderWavs
```

Listener samples are float32 files under `build/`. `script/RenderWavs` writes listening copies under `gen/wav/`. CUDA references are committed under `gen/cuda/`. Cross-platform agreement varies by scene; modal timing corrections and sensitive bubble dynamics prevent universal waveform identity. See [the comparison contract](VALIDATION.md#waveblender).

## SonicRadiation

This independent paper implementation couples a time-domain boundary-element solver to an FDTD transport grid. It supports closed rigid modal bodies. Water, speakers, point impulses, occluders and thin shells use WaveBlender instead.

```sh
(cd build && ./AcousticSolver --radiation --seconds 0.3 ../config/WineglassTap.json)
script/ValidateRadiation
```

`build/radiation/` contains `<output>_grid.bin`, `<output>_eq5.bin` and JSON metadata for grid sampling and retarded-potential evaluation. Motion rebuilds geometry-dependent data between simulation epochs.

## Room acoustics

Explicit scenes use converted [PFFDTD](https://github.com/bsxfun/pffdtd) voxelizer data. `ConvertRoomScene` writes the configuration and boundary, material, source and receiver arrays under `Scenes/`.

```sh
script/ConvertRoomScene RoomChurch
(cd build && ./AcousticSolver --room --seconds 0.2 ../Scenes/RoomChurch/config.json)
script/ValidateRoom
script/RenderRoomWavs
```

The implicit solver consumes a PFFDTD `model_export.json` directly:

```sh
(cd build && ./AcousticSolver --implicit-room /absolute/path/to/model_export.json --h 0.02 --seconds 0.1)
build/RoomTest --implicit
```

Implicit defaults reproduce the paper's Fig. 6 numerical configuration. Optional zero-mean projection is off by default. Explicit walls support parallel-LRC materials; implicit walls use real admittance.

Receiver arrays and sample-rate metadata are written under `build/room/`. `RenderRoomWavs` applies reference post-processing and writes normalized 48 kHz WAVs under `gen/wav/room/`.

## Immersed boundaries

JSON scenes define the medium, grid, PML, static sphere/square/mesh surfaces, a Gaussian volume-velocity source and receivers. Surfaces can be rigid, pressure-release, rational impedances/admittances or transmitting barriers. Use the room solver for enclosure walls.

```sh
build/AcousticSolver --immersed config/ImmersedSphere.json
build/AcousticSolver --immersed config/ImmersedBarrier.json --seconds 0.003 --output short_barrier
script/ValidateImmersed
```

Outputs are `build/immersed/<output>.bin` with adjacent JSON metadata. Samples are float32, interleaved by receiver within each timestep. Dense surface matrices impose quadratic memory cost in the patch count; moving surfaces are unsupported.

## Curved tetrahedral DG

Rigid-wall acoustics use FP32 propagation with conservation-preserving weight-adjusted mass inversion on curved tetrahedra. Geometry and operators are prepared in FP64.

```sh
build/DgReference prepare build/dg-inputs
build/DgTest build/dg-inputs/reduced/degree6/q12 build/dg-metal-check
```

See [DG usage and formats](src/Dg/README.md) for preparation, outputs, independent references and CUDA comparisons. Mesh loading, source injection and frequency-dependent walls are not integrated into a native scene runner.

## Nonlinear plates

Positioned strikes excite nonlinear plate vibration, with frequency-dependent damping, SAV drift regulation and multiple displacement pickups. This is an offline vibration renderer; acoustic radiation is a separate coupling stage.

```sh
build/PlateSolve config/PlateSmall.json
build/PlateSolve config/PlateCrash.json --output build/plate/crash
build/PlateSolve config/PlatePhysical.json --output build/plate/physical
script/ValidatePlate
```

Scaled configurations use `kappa`, `ratio`, `sigma0`, `sigma1` and `lambda`; coordinates lie in `[0, sqrt(ratio)] × [0, 1/sqrt(ratio)]`. `PlatePhysical.json` demonstrates material/dimension inputs, coordinates in metres and forces in newtons. Each strike has a position, amplitude, start, duration and cosine `type` (1: half cosine; 2: raised cosine). `repeat_count` repeats the excitation cycle while preserving state; `seconds` and `--seconds` specify each cycle's duration. An empty strike list with `initial_amplitude` excites the first mode.

`use_exact: false` selects standard linear coefficients; `norm_type` selects regulator norm 1 or 2. Missing/null `cutoff_hz` selects the stability limit. `oversampling` defaults to 1.5; `excitation_scale` and `output_scale` set additional gains. Metadata includes a rerunnable `resolved_config`.

Nonlinear runs automatically select dense or FFT Metal, both with 256-bit arithmetic and MPFR preparation. `--metal`, `--fft` and `--cpu` force a backend; linear automatic runs use CPU FP64. FFT evaluates the same transforms and is not necessarily faster.

Outputs include float64 pickup samples (`.bin`), six energy/work/drift diagnostics (`.diagnostics.bin`), metadata and final state (`.json`), and a normalized WAV. `--states` records every modal state; `--no-wav` omits WAV output. Physical pickups measure metres, while modal state and diagnostics retain scaled units.

Long sensitive trajectories can diverge from the authors' FP64 waveform; see [plate validation and limits](VALIDATION.md#nonlinear-plates). Initialize optional upstream comparison code with `git submodule update --init external/fa2026`.

## Nonlinear stiff strings

Port of [Russo, Ducceschi and Bilbao (2026)](https://doi.org/10.1007/s11071-026-12708-0),
with seven scalar auxiliary variable (SAV) formulations: geometric Form A
split/unsplit, geometric Form B split, cubic split/unsplit, and Kirchhoff–Carrier
split/unsplit. Simply supported strings start from a first mode or a raised-cosine
pluck; geometric models also evolve longitudinal displacement.

```sh
build/StringSolve config/StringGeometric.json
build/StringSolve config/StringCubic.json
build/StringSolve config/StringKirchhoffCarrier.json
```

`variant` selects `ge-a-unsplit`, `ge-a-split`, `ge-b-split`, `cubic-unsplit`,
`cubic-split`, `kc-unsplit`, or `kc-split`. `integrator` defaults to `"sav"`;
`"reference"` selects Verlet for geometric models or energy-preserving schemes
for cubic/Kirchhoff–Carrier. Both use Metal paired-float arithmetic by default;
`--cpu` selects FP64 and can be faster for a single string.

Configurations accept `seconds`, `oversampling` (relative to 44.1 kHz),
`initial_amplitude` in metres, `initial_condition` (`mode` or `raised-cosine`),
`initial_width` as a fraction of the interior grid, and `damping`. Physical inputs
are `density`, `tension`, `radius`, `young_modulus`, and `length` in SI units.
`loss: [[0, 15], [1000, 10]]` specifies frequency/T60 pairs in Hz/seconds.
`sample_rate` can replace `oversampling`. `potential_shift`, `longitudinal_grid`,
`stability_margin`, and explicit `intervals` expose the paper's numerical choices;
changing them can affect convergence. Omitted settings follow the upstream variant
presets, which can use very high sample rates. The supplied examples use lower
explicit rates. `nonlinear: false` provides a linear limit for cubic/Kirchhoff–Carrier.

The solver adjusts the grid and timestep as upstream does. Metadata records both
its rounded bookkeeping sample rate and the physical rate `1/dt`. Outputs under
`build/string/` include float64 displacement samples (`.bin`), six diagnostics
(`.diagnostics.bin`), configuration/final-state metadata (`.json`), and a normalized,
low-pass-resampled 48 kHz listening WAV. Geometric output has transverse and
longitudinal channels; other models have one channel. `--states` records every
interior-node state, `--no-wav` skips listening output, and `--output PREFIX` sets
the destination. Radiation and contact/bowing excitation are outside this model.
See [string validation and numerical limits](VALIDATION.md#nonlinear-stiff-strings).

## Acoustic Reliefs

Generate and optimize a heightfield with the Metal acoustic solver and an MPS appearance objective. A target image is required; completed runs produce an OBJ and resumable checkpoints. See [setup, rendering and upstream comparison instructions](script/reliefs/README.md).
