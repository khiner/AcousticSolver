# Validation

Run commands from the repository root after building. The checks below distinguish
analytic accuracy, agreement with an independent implementation, and repeatability
against saved outputs. Byte-identical reruns establish reproducibility, not physical
accuracy. Generated reports record the settings and tolerances used.

## WaveBlender

```sh
script/FetchScenes CupPhone
script/ValidateGolden CupPhone
script/ValidateGolden --score CupPhone
```

Omit scene names for the complete suite. CUDA listener references are committed
under `gen/cuda/<scene>/<sample-rate>/`; scene assets must be fetched separately.
Scoring uses existing `build/*.bin` files with `--score` and records results in
`gen/validation.json`. Only compare captures at the same simulation/sample rate.

Cross-platform CPU preparation and nonlinear bubble dynamics can accumulate
roundoff differences. The checker reports waveform, envelope and spectral
agreement separately. The native modal oscillator fixes a duplicated batch-boundary
step in upstream; the committed Modal-scene goldens retain that bug and therefore
support envelope rather than waveform agreement. Other deliberate differences
include per-component fresh-cell solves, ordered overlapping source writes,
precomputed bubble inverses, and retimed scene configurations. `ACOUSTIC_GLOBAL_LSTSQ=1`
selects the upstream global fresh-cell solve for diagnostic comparisons.

## SonicRadiation

```sh
build/RadiationTest --gate
script/ValidateRadiation
script/ValidateRadiation --exact
script/ValidateRadiation --score
```

The analytic ladder checks free-field behavior, boundaries, stability and motion.
Scene regressions exercise the mesh, source and geometry-epoch paths against
`gen/radiation/`. Default scoring requires 90 dB SNR; `--exact` requires identical
bytes. `--score` reads existing results. This is an independent implementation;
no public upstream executable supplies a full trajectory reference.

Material numerical departures include a larger near set and filtered grid feedback
for stability, a diagonal boundary solve, remeshing to the transport grid, and
quantized/truncated history weights. Both grid-sampled and retarded-potential
outputs are tested. The method's scope is closed rigid modal bodies, not all
WaveBlender scene types.

## Room acoustics

```sh
build/RoomTest
script/ValidateRoom
script/ValidateRoom --exact
script/ValidateRoom --score
build/RoomTest --implicit
```

Explicit checks cover analytic modes, boundary losses, energy, stability and
repeatability. PFFDTD comparisons use the same converted grid, boundary data,
sources and receiver clocks. Native committed captures under `gen/room/` provide
change detection; agreement with native bytes is separate from CUDA-reference
accuracy. `--exact` requires identical native output bytes.

The implicit ladder checks dispersion, voxelization, boundaries, stability and
optional mean projection. It uses the production 27-point method, which consumes
PFFDTD model exports directly. Explicit frequency-dependent LRC walls and implicit
real-admittance walls are different supported boundary models. Mean projection is
off by default and must be recorded when used.

Room WAVs use differentiated-pressure receiver data, reference integration/high-pass
filtering, resampling and air absorption. Raw samples and processed WAVs are not
interchangeable numerical references. Finite record lengths can truncate decay;
a rigid room's truncated tail must not be interpreted as physical damping.

## Porous room elements

```sh
script/ValidatePorous
script/ValidatePorous --cpu
script/ValidatePorous --demo
script/ValidatePorous --band  # extended frequency sweep; requires matplotlib
```

The focused checks cover material fits, transparent interfaces, box/OBJ
classification, rigid-room attenuation, FP64/Metal agreement and split-run
repeatability. Metal receiver relative error must remain below `2e-4` against the
FP64 discrete reference. The source starts from rest; arbitrary nonzero porous
initial fields require consistent auxiliary states.

Production-solver slab checks compare complex reflection and transmission with
an independent continuum transfer-matrix calculation. A 10 cm absorber with
flow resistivity 10000 Pa·s/m² is tested at approximately 1011 Hz, at normal and
45° incidence, both freestanding and backed by an approximately 20 cm air gap.
The exact voxelized gap is used in each comparison. Refining spacing from 2 cm
to 1 cm must reduce errors, with fine-grid complex-amplitude errors below `0.03`.
Stationarity and spatial wave-fit residuals must be below `0.001`, with
decomposition condition below 10. `build/porous/slabs.json` records geometry,
coefficients, errors and timings.

`--band` extends this to 20 frequencies spanning 20–4000 Hz, at normal and
near-45° incidence (43.86–46.23°), with both backing configurations and two grid resolutions.
Transverse mirror-mode dimensions are rounded to the grid; the exact angle and
backing gap enter the analytical comparison and are recorded for every case.
Runs last at least 96 cycles and 200 ms. Fine spacing decreases from 25 mm at
low frequencies to 1.25 mm at the upper band edge.

Each pair must reduce complex reflection/transmission errors under refinement,
with fine-grid errors below `0.03`. Absorption error must be below `0.05`;
impedance error is `|z-z_ref|/max(1,|z_ref|) < 0.05`, where `z = Z/(rho*c)`.
The same `0.001` stationarity and spatial-residual limits apply. The wave
decomposition condition limit is 100 because the fixed probe span covers only
a small part of a wavelength at 20 Hz.

`build/porous/band.json` records the results, hardware and executable/source
hashes; `band-backed` and `band-freestanding` PNG/SVG plots show the comparisons.
This validates sampled frequency-dependent scattering with the production
Metal solver. It does not reproduce the paper's large finite-aperture pulse
experiment or its full-audio-band room benchmark.

Material preparation requires stable real poles, positive K/R relaxation weights,
and sampled passivity of Q/s. Fits are checked densely against the frequency-domain
Allard–Champoux model. These checks and bounded post-excitation runs do not prove
stability for arbitrary fitted materials, grids or durations.

The demo produces shared-gain listening WAVs and `build/porous/demo.json` with
render time, allocated Metal buffer bytes and late pressure RMS. Render time
includes GPU initialization, stepping and receiver extraction but excludes file
writing and WAV processing; buffer bytes exclude driver/pipeline allocations.
Late pressure RMS is a listening comparison, not a stored-energy or RT60 estimate.
The 4 cm demo grid is intended for low-frequency room effects; its 4 kHz material
fit limit does not establish 4 kHz spatial accuracy.

## Immersed boundaries

```sh
build/ImmersedTest --gate
script/ValidateImmersed
script/ValidateImmersed --exact
script/ValidateImmersed --score
```

The ladder checks free-field and analytic sphere solutions, energy, boundary
limits and stability. Scene regressions use `config/ImmersedSphere.json` and
`config/ImmersedBarrier.json` against `gen/immersed/`. Default scoring requires
90 dB SNR; `--exact` requires identical bytes. Longer figure and stability runs
are available through `ImmersedTest --energy` and `--soak`.

The implementation uses an outer split PML and deterministic equal-area sphere
sampling, rather than reproducing every discretization choice in the papers.
It supports static exterior surfaces and barriers, not moving surfaces or
conservative enclosure walls. Surface matrices require quadratic memory in patch
count, and ill-conditioned limiting projections are rejected at setup.

## Curved tetrahedral DG

```sh
build/DgReference prepare build/dg-inputs
build/DgReference check build/dg-reference-checks
build/DgReference validate build/dg-metal build/dg-inputs
```

The independent FP64 reference and analytic rigid-cylinder solution test geometry,
mass inversion, spatial operators and propagation. The regression suite covers
both fixture meshes, timestep refinement, larger/odd working sets, alternate mass
quadrature, invalid inputs, and repeated state/receiver bytes. Main gates are:

- Analytic terminal field error below `1e-3`
- State, receiver, mass and scaled RHS errors against FP64 below `2e-5`
- Exactly zero constant-pressure RHS
- Modified energy at most `1.00001` times its initial value
- Physical mean drift below `1e-5` of initial acoustic RMS
- Bitwise-identical repeated terminal states and receiver records, with matching clocks

Mass-quadrature reductions additionally check positive geometry/masses, constant
moments and spectrum. The stock degree-8 cylinder remains outside the validated
FP32 scope. These fixture checks do not establish a timestep bound for arbitrary
meshes. See [formats and independent reference commands](src/Dg/README.md).

The [upstream CUDA comparison](src/Dg/README.md#upstream-cuda-comparison) uses matched
meshes, initial states and clocks, but different spatial formulations. Curved-mesh
waveforms need not match exactly; assess conservation and independent FP64 errors
separately from runtime.

## Nonlinear plates

```sh
build/PlateTest --gate
script/ValidatePlate --output build/plate-validation/dense
script/ValidatePlate --fft --output build/plate-validation/fft
script/ValidatePlate --score --fft --output build/plate-validation/fft
```

Initialize `external/fa2026` for upstream comparisons. The wrapper runs the pinned,
unmodified Python code in an isolated environment and records revision, dependency,
configuration and output hashes. Short comparisons use upstream PSTD and selected
modal-tensor cases. `--score` checks saved artifacts and implementation identity;
`--reuse-reference` reuses only verified upstream captures.

Coverage includes transforms, nonlinear gradients, padding, frequency/decay,
energy/work balance, both regulator norms, initialization, repeated strikes,
physical scaling and both Metal backends. [ValidatePlate](script/ValidatePlate)
defines per-case tolerances, including wider limits for the symmetric regulated
initial-state case whose velocity-sign regulator amplifies roundoff.

### High-precision Metal accuracy

Long-trajectory checks use an independent high-precision implementation of the
discrete equations. The 3-second unregulated small crash is validated against a
converged reference. This establishes neither upstream FP64 waveform parity nor
continuum accuracy.

```sh
# Generate a reference once; the small crash takes about half an hour on M5 Max.
script/PlatePrecisionRender --output build/plate-reference/crash
# Reuse it for current Metal validation.
script/ValidatePlateAccuracy --reference build/plate-reference/crash --output build/plate-accuracy/crash
script/ValidatePlateAccuracy --score --reference build/plate-reference/crash --output build/plate-accuracy/crash
```

The convergence gate verifies reference hashes and agreement, then requires `1e-8`
relative agreement for waveform, complete modal history, 10 ms modal blocks,
final state, six diagnostics and STFT magnitude. It also checks energy/work balance
within `2e-9` of peak energy. It checks automatic Metal and
explicit FFT selection and records source/executable identities. `--steps` on the
reference generator provides a bounded smoke, not full convergence evidence.

`PlatePrecisionRender --case NAME` selects any fixture in `config/plate/`, including
the energy, repeated-strike, amplitude, stock and decay presets. `--config PATH`
accepts scaled nonlinear 44.1 kHz configurations with exact linear coefficients,
including `config/PlateLarge.json`. Give each reference its own `--output` directory,
then pass that directory to `ValidatePlateAccuracy --reference`. Availability of a
fixture does not establish full-duration convergence for it.

Two 9-second repeated-strike fixtures lack converged full-duration references,
and the 400:1 aspect-ratio stress case exceeds its CPU potential-error gate.
Neither is covered by the small-crash validation claim.

## Nonlinear stiff strings

```sh
build/StringTest --metal
# Dependencies for comparisons with the pinned upstream MATLAB scripts:
brew install octave
python3 -m pip install numpy soundfile
git submodule update --init external/nLinStringsConv-SAV
script/ValidateString
script/ValidateString --long --output build/string-validation-long
```

`StringTest` checks gradients, conservation/damping, analytic modes, timestep
refinement and CPU/Metal agreement.
`ValidateString` compares all seven variants and both integrators against Octave:
grids, clocks, displacement, full node states and available diagnostics. Regular
captures are 2 ms; `--long` extends them to 60 ms, while small-shift smoke cases
remain capped at 0.5 ms. `--cpu-only` skips Metal; `--reuse-reference` verifies
configuration and hashes before reusing captures. Reports contain measured errors
and the enforced tolerances.

For a single reference capture, run `script/RunStringReference CONFIG --output PREFIX`.
It preserves the upstream submodule and saves its generated adapter and provenance.
`--max-steps` and `--max-points` override its size limits for large upstream presets.

Geometric Form A is sensitive to timestep, grid and potential shift. The supplied
Form B example uses a longitudinal grid; a transverse bound alone may not resolve
longitudinal motion. `--stress --long` exercises sensitive small-shift Form A cases
that can fail waveform agreement, including sensitivity within upstream itself.
Energy conservation alone does not establish trajectory or continuum accuracy.

Upstream GE A unsplit energy uses stale states, and GE reference energy arrays are
unfilled; those energy columns are excluded from parity checks. The upstream
unsplit SAV damping update also lacks the split scheme's energy-balance guarantee.
The native `nonlinear: false` limit is checked analytically because upstream's
corresponding branches are inconsistent.

### Paper experiments

```sh
python3 -m pip install numpy matplotlib
python3 script/string/PaperOde.py
python3 script/string/PaperStrings.py --workers 4
```

These CPU sweeps reconstruct Figures 1–7 and compare with Tables 2–3. The full
string sweep takes tens of minutes on an M5 Max. Results, figures and provenance
are written under `build/string-paper/{ode,strings}/`. String runs use verified
caches; `--plot-only` regenerates figures from saved results. Use ODE `--smoke` or
lower string `--max-exponent`/`--reference-exponent` values for reduced checks.

Some geometric cases terminate numerically, and the published table values are
not reproduced exactly. Reports retain failures, discrepancies and interpretation
choices, including initialization, sample timing and the ODE force-sign correction.
Completing the sweeps does not establish convergence of every formulation.

## Coupled plates and air

```sh
git submodule update --init external/ness
script/ValidateCoupled
script/ValidateCoupled --demo  # one second each of supported/free plate and drum audio
```

The validator compares Metal float-pair stepping against pinned NESS FP64 for
the original discretization, and a separate FP64 sparse direct solve for explicit
rectangular edge conditions. Short cases cover nonlinear/linear motion, silence,
sympathetic plates, reduced air loading, repeated strikes with a reflecting
exterior, two circular drum heads with a rigid shell, all three rectangular edge
conditions, and a scene mixing boundary types, grid sizes, linear/nonlinear
panels and coupling iteration counts. It checks that
nonlinearity and air loading change the motion and that unstruck surfaces respond.
Each receiver trace and the complete final air/panel states must have relative
L2 error below `1e-4`; the reports preserve measured errors and source/binary hashes.
WAV checks cover rate, shape, finite samples and normalization. Two malformed
configurations must be rejected. Free-edge strikes must move boundary nodes;
clamped and simply supported boundary displacements must stay zero.

`CoupledBoundaryTest` also checks mass-weighted stiffness symmetry, the three
free-plate rigid modes, nonlinear polarization, lossless nonlinear energy, and
the plate/air interpolation work identity. Fundamental frequencies on 8-, 16-
and 24-interval square grids must converge toward the analytic simply supported
value and the clamped/free Rayleigh–Ritz references in
[Leissa, NASA SP-160, Tables 4.22 and 4.67](https://ntrs.nasa.gov/citations/19700009156),
within 1.2% on the finest grid. The nonlinear energy check allows relative drift
below `1e-8` and requires at least 1% of energy to enter the nonlinear term.
The Airy boundary treatment follows
[Thomas and Bilbao, Eqs. 21 and 33](https://doi.org/10.1016/j.jsv.2008.04.014).

CPU/Metal comparisons share prepared coefficients; the mode/refinement and
energy checks provide additional evidence for the new boundary operators.
Pressure is the centered velocity-potential derivative multiplied by air density;
this adds the `rho/2` factor omitted by NESS's `OutputPressure` convention.
The build applies circular-mask bounds guards and initializes an unused plate
wave-speed argument in a generated NESS source copy; the submodule stays unchanged.
The frequency checks establish refinement behavior for isolated plates, not
convergence of every coupled nonlinear trajectory or reproduction of every paper figure.

## Acoustic Reliefs

The [sanity and upstream comparison commands](script/reliefs/README.md) test Metal
pressure/height gradients against dense FP64 and compare matched mesh/target states
with the authors' implementation. Acoustic and appearance objectives are assessed
separately. Optimization trajectories may differ under floating-point rounding.

## Reproducing results

Use the same input, discretization, source clock and output precision when comparing
solvers. Preserve generated metadata and report hashes. `--score` avoids rerendering
where supported. The Radiation, Room and Immersed validators' `--update` options
replace committed regression references; they are not ordinary validation runs.

For timing, record hardware, setup/file-output exclusions, warmup, repeated runs and
competing machine load. Solver-specific benchmark tools are `RadiationBench`,
`RoomBench`, `BenchmarkPlate`, and the DG benchmark mode. Runtime Metal shaders are
loaded from the source checkout recorded at build time; separate binaries from one
checkout do not provide independent shader baselines.

## Viscothermal tubes

```sh
git submodule update --init external/openwind
python3 -m venv --system-site-packages build/tube-reference/venv
build/tube-reference/venv/bin/python -m pip install numpy scipy matplotlib h5py soundfile
build/tube-reference/venv/bin/python script/ValidateTube --demo
```

The validator executes pinned OpenWind v0.12.4 with independently assembled
Gauss–Lobatto operators and the same explicit SI material inputs, geometry,
endpoint conditions and excitation. It compares native preparation coefficients,
FP64 receiver pressures, final acoustic/memory states, and discrete energy against
the independently implemented OpenWind solver.

Cases cover tapered tubes, polynomial degrees 1, 3 and 10, loss approximations with
0, 4, 8 and 16 terms, closed/open/matched/resistive ends, excitation from either
end, silence, lossless conservation and passive decay, including a larger mesh.
`--demo` includes the five-second README example. Errors must remain below `1e-7` relative L2; passivity is checked
separately after excitation ends. Invalid geometry, approximation order and CFL
settings must fail before stepping.

A cylinder refinement check compares the damped Laplace transform of the rendered
input pressure/flow with the independent Zwikker–Kosten Bessel-function impedance
over 100–3000 Hz. The final mesh must reach 2% relative error and improve on the
coarse mesh. Exponential windowing limits truncation of the impulse response.
This checks the continuum approximation and discretization together.

`build/tube-validation/` contains raw native/reference traces, listening WAVs,
impedance data, logs and a summary with source/executable hashes. `--output` and
`--executable` select another output directory or build. The model equations and
source implementation are compared; the paper's exact trumpet plots and human
perceptual equivalence are not claimed.
