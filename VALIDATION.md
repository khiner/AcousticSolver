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
physical scaling, full-width arithmetic and FFT scratch storage. The numerical
ladder passes 115 checks; each short Metal suite passes 866 comparisons. Usual
waveform/modal tolerances are `2e-8`, with `4e-8` for final states. The symmetric
regulated initial-state case uses `1e-5`/`2e-5` on affected waveform/state checks
because its velocity-sign regulator amplifies roundoff between formulations.
The thresholds are enforced in [ValidatePlate](script/ValidatePlate).

### High-precision Metal accuracy

For the full 3-second unregulated small crash, dense and FFT Metal match an
independent 214/427-bit converged reference at exported binary64 precision.
Waveform and complete modal history are byte-identical; diagnostic differences
are limited to signed zeros. This reference is an independent implementation of
the discrete equations, not upstream Python executed at higher precision.
Convergence is distinct from upstream FP64 waveform parity and continuum accuracy.

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
then pass that directory to `ValidatePlateAccuracy --reference`. These replace the
old long-duration FP64 waveform gates; availability of a fixture does not establish
full-duration convergence for it.

Two 9-second repeated-strike fixtures still lack converged full-duration references.
A 400:1 aspect-ratio stress case retains a CPU potential error of `2.06e-12` against
a `2e-12` gate. Neither result is covered by the small-crash claim. The renderer
produces displacement; acoustic radiation and audio callbacks are outside scope.

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
