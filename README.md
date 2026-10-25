# AcousticSolver

Offline acoustic solvers for Apple Silicon, implemented in C++ and Metal.

| Solver | Use | Scope |
| --- | --- | --- |
| [WaveBlender](https://github.com/kangruix/WaveBlender) | Sound from animated scenes | Modal bodies, bubbles, speakers, impulses and occluders |
| [SonicRadiation](https://arxiv.org/abs/2508.08775) | Exterior sound radiation | Closed rigid modal bodies, including motion |
| [Room acoustics](https://www.pure.ed.ac.uk/ws/files/22154168/fv_genimp_final_r3.pdf) | Room impulse responses | Cartesian/FCC FDTD with rigid or frequency-dependent walls |
| [Porous room elements](https://www.dafx.de/paper-archive/2023/DAFx23_paper_46.pdf) | Absorber panels, air gaps and porous partitions | FCC propagation through rigid-frame equivalent fluids |
| [Implicit room acoustics](https://doi.org/10.1121/10.0036229) | Low-dispersion room simulation | 27-point scheme with real-admittance walls |
| [Immersed boundaries](https://doi.org/10.1121/10.0020635) | Exterior scattering and transmission | Static surfaces, impedances and finite barriers |
| [Curved tetrahedral DG](src/Dg/README.md) | Rigid-wall wave propagation | Prepared meshes and operators; no native scene/source integration |
| [Coupled elastic–acoustic waves](src/Elastic/README.md) | Transmission, structural radiation and fluid-filled elastic objects | General conforming tetrahedral fluids/solids, curved geometry, heterogeneity and anisotropy |
| [Viscoelastic waves](https://doi.org/10.1016/j.camwa.2021.08.003) | Material damping in plastics, elastomers and layered solids | Generalized Zener relaxation, frequency-dependent attenuation and dispersion, coupled to fluid acoustics |
| [Nonlinear plates](https://arxiv.org/abs/2608.06139) | Plate vibration and crash sounds | Simply supported rectangular plates; displacement output |
| [Nonlinear stiff strings](https://doi.org/10.1007/s11071-026-12708-0) | Plucked-string vibration | Geometrically exact, cubic and Kirchhoff–Carrier models; displacement output |
| [Coupled plates and air](https://www.ness.music.ed.ac.uk/wp-content/uploads/2013/06/dafx2013_submission_41.pdf) | Radiating panels, sympathetic vibration and drums | Nonlinear parallel plates/membranes with two-way 3D air coupling |
| [Viscothermal tubes](https://doi.org/10.1121/10.0005537) | Narrow pipes, tapered ducts and tube resonators | 1D pressure/volume-flow propagation with viscous and thermal boundary-layer losses |
| [Thermoviscous boundary elements](src/Thermoviscous/README.md) | Small cavities, gaps and viscous sound radiation | 3D harmonic acoustic, thermal and viscous modes; prescribed isothermal no-slip walls |
| [NumCalc](https://doi.org/10.1016/j.enganabound.2024.01.008) | Scattering, enclosures and prescribed-wall radiation | Interior/exterior BEM, Burton–Miller stabilization, admittance boundaries and multilevel FMM |
| [Acoustic Reliefs](script/reliefs/README.md) | Acoustic heightfield optimization | Differentiable boundary elements with an appearance objective |

See [validation](VALIDATION.md) for checks and numerical limits, and [third-party notices](NOTICE.md) for licensing and reference-code provenance.

## Build

Requires macOS on Apple Silicon, Xcode's Metal toolchain, CMake 3.24+, and a C++23 compiler. Homebrew LLVM is used for development. Boost, MPFR and GMP support plate preparation; Eigen is vendored and CMake downloads pinned metal-cpp headers. GCC supplies gfortran for the thermoviscous FMM library; initialize the submodules before configuring.

```sh
brew install cmake ninja llvm boost mpfr gmp gcc
git submodule update --init --recursive
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

## Porous room elements

Independent implementation of Jan W. Smits,
[*Efficient finite-difference room acoustics simulation incorporating extended-reacting elements* (DAFx 2023)](https://www.dafx.de/paper-archive/2023/DAFx23_paper_46.pdf).
Metal evolves pressure inside porous volumes and pressure/velocity at their
interfaces, including transmission, oblique incidence and air gaps.

```sh
(cd build && ./AcousticSolver --room ../config/PorousRoom.json)
script/ValidatePorous --demo
script/ValidatePorous --band  # 20–4000 Hz scattering validation and plots
```

[PorousRoom.json](config/PorousRoom.json) defines a rigid shoebox with a backed
panel and a freestanding partition. `room` sets dimensions and grid spacing in
metres, Courant number and duration. Sources and receivers use physical coordinates.
`porous_materials` specifies `flow_resistivity` in Pa·s/m² and optional `fit_hz`
limits (default 20–4000 Hz) for the Allard–Champoux model. `porous_volumes` assigns
a zero-based material index to each `min`/`max` box or relative-path `mesh` OBJ.
Meshes must be closed, connected, consistently wound and free of self-intersections;
self-intersections are not detected.

Porous volumes may meet rigid walls but must not overlap each other or locally
reacting wall nodes. Sources must be in scalar air, away from porous interfaces;
receivers must be in air. Existing converted FCC scenes also accept these material
and volume fields when `grid.origin` specifies the physical position of grid
index `(0,0,0)`. Cartesian porous scenes are unsupported.

Native shoebox outputs are float64 **pressure**, stored receiver by receiver under
`build/room/`, with field type, timings and Metal buffer bytes in adjacent JSON.
The demo writes treated/untreated 48 kHz WAVs under `build/porous/`, using the same
30–1000 Hz filter and shared gain. Use this renderer for native pressure output;
`RenderRoomWavs` expects the older differentiated-pressure scene format.

The fit band describes material approximation, not grid accuracy. Resolve the
shorter wavelength inside the porous medium and check spatial convergence.
This is a rigid-frame model: fabric motion, elastic foam frames and structural
vibration are outside its scope. See [porous validation](VALIDATION.md#porous-room-elements).

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

## Coupled plates and air

Metal port of the coupled model in Torin and Bilbao,
[*A 3D Multi-Plate Environment for Sound Synthesis* (DAFx 2013)](https://www.ness.music.ed.ac.uk/wp-content/uploads/2013/06/dafx2013_submission_41.pdf),
using the later [NESS implementation](https://github.com/Edinburgh-Acoustics-and-Audio-Group/ness)
for grid/operator preparation and an FP64 reference. Metal advances the air,
nonlinear plates, iterative stress solve and both directions of the coupling.

```sh
git submodule update --init external/ness
PATH="$(brew --prefix llvm)/bin:$PATH" script/Build
build/CoupledSolve config/CoupledPlate.json --output build/coupled/plate
build/CoupledSolve config/CoupledFreePlate.json --output build/coupled/free-plate
build/CoupledSolve config/CoupledDrum.json --output build/coupled/drum
script/ValidateCoupled --demo
```

The airbox is centered at the origin. Panels lie parallel to the XY plane, with
physical `center`, rectangular `size` or circular `radius`, `thickness`, `density`,
`young_modulus`, `poisson_ratio`, optional `tension` (N/m), `t60` and `sigma1`.
Dimensions are in metres and times in seconds. Set `nonlinear: false` per panel
for linear motion. Strikes specify a zero-based `panel`, normalized `position`
within its bounding rectangle, raised-cosine `force` in newtons, `start` and `duration`.
Receivers use physical coordinates. `drum_shell` connects two centered circular
heads of equal radius with a rigid cylindrical shell.

Rectangular panels accept `boundary: "clamped"`, `"simply_supported"`, or `"free"`.
These select transverse edge conditions; all three use traction-free in-plane
conditions, with zero Airy stress function and normal derivative at the perimeter.
Free edges retain their displacement nodes and corner/edge mass weights, including
rigid translation and tilt modes when tension is zero. A free plate cannot carry
prescribed uniform `tension`. Omitting `boundary` retains the original `"ness"`
discretization for reference comparisons; circular membranes use that discretization.
Different rectangular edge conditions can share one airbox.

`--seconds N` overrides duration; `--cpu` selects the FP64 reference; `--compare` renders both
backends and checks receiver traces and final fields. Outputs are native-rate
float32 `.wav` listening copies and float64 `.bin` samples interleaved by timestep:
air pressures in Pa, then each panel's center displacement in metres. WAVs contain
only the air channels, with one normalization gain recorded in `.json` alongside
resolved grids, backend, timing and reference revision. `.air.bin` stores final
air velocity potential; `.panels.bin` concatenates final panel displacements.
Comparison runs write `.upstream.wav`/`.upstream.bin` for NESS-only scenes or
`.reference.wav`/`.reference.bin` for scenes using explicit rectangular boundaries.
The latter reference uses a sparse direct plate solve and NESS air stepping.

Supported geometry is static and parallel to the XY plane.
Panels require two air cells of lateral clearance and four slices of separation.
The exterior uses NESS's absorbing boundary by default; `reflecting: true` selects
a zero-potential, pressure-release boundary. `lossless: true` disables structural
and air attenuation. Arbitrary plate rotations and contact are not implemented.

Resolution follows the requested sample rate and NESS stability grid. Limits are
four million air cells, 4096 points per panel, 512 MiB of prepared operators and
256 MiB of trace/excitation storage. The dense inverse preconditioner costs
quadratic storage per nonlinear panel. `solver_tolerance` (default `1e-9`),
`solver_iterations` (64), and per-panel `coupling_iterations` (5 rectangular,
8 circular) control the iterative solves. Metal uses float-pair arithmetic and
reports failure if PCG does not converge. The CPU reference can be faster on small
scenes, particularly the two-head drum. GPU work is submitted in bounded
batches, with panel-size-dependent PCG segments and concurrent panel solves.
Reported render times exclude operator preparation, shader compilation and file output.
Set `ACOUSTIC_PROFILE=1` to report GPU execution, submission gaps and host wait time.

## Viscothermal tubes

Port of Thibault and Chabassier,
[*Dissipative time-domain one-dimensional model for viscothermal acoustic propagation in wind instruments* (JASA, 2021)](https://doi.org/10.1121/10.0005537).
Native C++ prepares the spectral finite elements and diffusive loss coefficients,
then advances pressure, volume flow, and viscous/thermal memory states in FP64 on the CPU.
[OpenWind](https://gitlab.inria.fr/openwind/openwind) is pinned as an independent Python reference.

```sh
PATH="$(brew --prefix llvm)/bin:$PATH" script/Build
build/TubeSolve config/Tube.json --output build/tube/render
```

`bore` lists increasing `[position, radius]` pairs in metres, starting at zero.
Radius is linear between pairs. `elements_per_segment` gives the number of elements
in each interval; `order` selects polynomial degree 1–10. `loss_terms` accepts
0–8 or 16 auxiliary terms from the upstream tables; zero retains the lowest-order
loss model, while `lossless: true` disables both viscous and thermal losses.
`air` optionally specifies `density` (kg/m³), `sound_speed` (m/s), `viscosity` (Pa·s),
`conductivity` (W/(m·K)), `heat_capacity` (J/(kg·K)), and `heat_ratio`.
Defaults approximate air at 20 °C and are recorded in the output metadata.

The two `ends` accept `"closed"`, `"open"` (zero pressure), `"matched"` (local
characteristic impedance), or a nonnegative resistance in Pa·s/m³.
Each raised-cosine pulse specifies `start`, `duration`, peak volume `flow` in m³/s,
and `end` (0 or 1). Positive flow enters the tube. An open end cannot accept a
flow source. `receivers` are positions along the tube in metres.

`sample_rate` is the actual simulation and WAV rate; there is no implicit resampling.
The renderer rejects rates below a conservative CFL bound, reporting the required
minimum. Increase the rate or reduce mesh density/order. `--seconds N` overrides
duration. The validator checks the native implementation against OpenWind.

Outputs are interleaved float64 pressure `.bin` samples in Pa, discrete energy
`.energy.bin` in joules, final `.state.bin`, a normalized 24-bit listening `.wav`,
and `.json` with physical inputs, mesh, normalization gain and timing.
Samples are recorded after each step, beginning at `1/sample_rate`.
The final state stores each pressure node as `[P, P0, Pi...]`, followed by each
discontinuous flow node as `[V, Vi...]`. Pressure states use Pa; flow states use m³/s.

The model assumes rigid axisymmetric walls and approximately plane acoustic waves.
It does not generate turbulent/whistling excitation or simulate mean flow, wall
motion, nonlinear propagation, or the exterior radiation field. Endpoint resistance
is a local termination model. The mesh is limited to 4096 pressure nodes and 8192
flow nodes. Reported render times exclude preparation and file output.
See [validation](VALIDATION.md#viscothermal-tubes) for upstream comparisons.

## Thermoviscous boundary elements

[Preuss et al. (2023)](https://doi.org/10.1142/S2591728523500159) resolve viscous and thermal losses on surface meshes, using a nested Schur-complement solve. The FP64 CPU implementation supports interior cavities and exterior radiation, linear/quadratic triangles and quads, dense/FMM/H-matrix acoustic operators, and prescribed wall motion.

```sh
script/PrepareThermoviscous build/thermoviscous/sphere.json --radius .01 --frequency 1000
build/ThermoviscousSolve build/thermoviscous/sphere.json build/thermoviscous/result.json
script/ThermoviscousDemo  # two-second steady-state listening sample
script/RenderThermoviscous build/thermoviscous/sphere.json  # broadband pulse or --input WAV
```

See [mesh import, material parameters and numerical limits](src/Thermoviscous/README.md). `ThermoviscousDemo` renders a steady tone; `RenderThermoviscous` sweeps frequencies and convolves the receiver response with an excitation.

## Acoustic scattering and radiation

[NumCalc (Kreuzer et al., 2024)](https://doi.org/10.1016/j.enganabound.2024.01.008) solves scattering and radiation from surface meshes with pressure, velocity or frequency-dependent admittance boundaries. The native integration retains the author's quadrature and dense/single-level/multilevel FMM formulations, with Accelerate CPU algebra and an optional compensated Metal operator backend.

```sh
script/PrepareNumCalc build/numcalc/sphere.json
script/RunNumCalc build/numcalc/sphere.json --output build/numcalc/sphere
script/RenderNumCalc build/numcalc/sphere.json --output build/numcalc/audio
```

See [geometry, boundary data, performance and numerical limits](src/NumCalc/README.md). `PrepareNumCalc --scene enclosure` generates a vibrating enclosure; `--scene duct` generates a terminated interior duct. Audio rendering accepts mono WAV excitation with `--input`.

## Acoustic Reliefs

Generate and optimize a heightfield with the Metal acoustic solver and an MPS appearance objective. A target image is required; completed runs produce an OBJ and resumable checkpoints. See [setup, rendering and upstream comparison instructions](script/reliefs/README.md).

## Coupled elastic–acoustic waves

[Guo, Acosta and Chan (JCP 2020)](https://doi.org/10.1016/j.jcp.2020.109632)
provides the coupled DG propagation model for general fluid and solid volumes.
`ElasticSolve` uses GPU-resident paired-float propagation with fixed timesteps
and an Accelerate FP64 reference (`--cpu`), with tagged curved tetrahedra, isotropic or
anisotropic elasticity, spatially varying materials, point sources and receivers.
The GPU renderer requires macOS 26 or newer and a Metal 4-capable Apple GPU.

```sh
python3 script/ElasticDemo sphere build/elastic-demo/sphere
build/ElasticSolve build/elastic-demo/sphere/config.json --prepare
build/ElasticSolve build/elastic-demo/sphere/config.json --output build/elastic-demo/sphere/render
```

Preparation requires NumPy and Gmsh. The example requests one second; `--seconds N`
overrides it. This explicit volumetric solver can be much slower than rendered audio.
See [inputs, outputs and physical limits](src/Elastic/README.md), including the
reflection limits of its local absorbing boundary.

## Viscoelastic waves

[Shukla, Chan and de Hoop (2021)](https://doi.org/10.1016/j.camwa.2021.08.003)
adds solid relaxation to `ElasticSolve`: anisotropic, heterogeneous materials can
have multiple Zener mechanisms, with frequency-dependent attenuation and dispersion.
Material dissipation is reported separately from interface and boundary loss.

```sh
python3 script/ElasticDemo viscoelastic build/elastic-demo/viscoelastic
build/ElasticSolve build/elastic-demo/viscoelastic/config.json --output build/elastic-demo/viscoelastic/render
```

This two-second example uses illustrative relaxation times. See
[material inputs and stiffness conventions](src/Elastic/README.md#solid-relaxation).
`build/ElasticTest --viscoelastic` runs the focused analytical and CPU/Metal checks,
including an independent comparison with SeisSol's attenuation coefficients.

## Elastic–acoustic demo videos

Two-second simulations of the coupled model from
[Guo, Acosta and Chan (JCP 2020)](https://doi.org/10.1016/j.jcp.2020.109632),
with synchronized pressure audio and field visualization at real-time playback speed.

| Scene | MP4 | Audio |
| --- | --- | --- |
| Soft elastic sphere in air | [Watch](media/elastic/sphere.mp4) | Mono: outside air pressure |
| Water-filled acrylic shell | [Watch](media/elastic/container.mp4) | Stereo: outside air (left), inside water (right) |

Audio uses a fixed gain per channel over the whole clip. The videos show instantaneous
fluid pressure and solid speed at element centres on a mesh slice, with fixed logarithmic
colour scales. The adjacent JSON files record scene settings, gains and render timings.
