# Validation

Run commands from the repository root after building. Reports record settings,
tolerances and provenance. Analytic checks assess numerical accuracy; independent
implementations test agreement; saved captures detect regressions. Identical bytes
alone do not establish physical accuracy.

Where supported, `--score` checks existing captures and `--exact` requires identical
bytes. The Radiation, Room and Immersed validators' `--update` options replace
committed references and are not ordinary validation runs.

## Differentiable physical-model ports

The validation commands select isolated Python environments and install missing pinned dependencies. They do not download author datasets or contact remote compute. Build the native targets with `script/Build` before validating Stable Modal Synthesis. Use `external/` submodules only for deliberate author-code comparisons; retained regression fixtures do not require them.

### DiffImpact

```sh
script/ValidateDiffImpact
```

The [accepted local baseline](tests/fixtures/diffimpact/manifest.json) covers forward values and vector-Jacobian products for the DSP components, complete hammer/separation graphs, MFCCs, GRU, encoder/decoder and property networks. Independent encoder checks instantiate the actual released classes with identical weights. They also cover the complete audio-to-synthesis graph, every trainable parameter, spectral-loss gradients and three Adam updates including both moment tensors. These bounded comparisons use explicit ASMR wiring and the documented batch-index correction; float32 phase and spectral-loss sensitivity prevent a promise of identical cross-framework training trajectories. The property-network fixtures are local regression checks, not upstream equivalence checks. Routine checks use the accepted native fixture, plus analytic, batch-isolation, padding, split-isolation, source-assignment, stem-composition and checkpoint-continuation checks:


The retained hammer and separation checkpoints cover completed native 10,000-step and 6,000-step schedules. These are adapted native fits, not author checkpoints. The original ASMR selection/settings, property-network checkpoints, external baseline training and human study remain unreproduced; curated ASMR recordings are explicitly an adaptation. See [model and usage](script/diffimpact/README.md).

### DiffSound

```sh
script/ValidateDiffSound
```

Retained author-code fixtures cover synthesis and parameter gradients, recorded-audio damping controls, noise, spectral/transport objectives, both FEM orders, tetrahedral clipping, shell derivatives, the neural SDF and acoustic transfer. Additional checks cover finite-difference eigenvalue derivatives and exact interrupted/resumed optimization across the loss-stage reset. The author reference runs on CPU with documented dependency/device substitutions; it is not a CUDA comparison. Runtime fitting and validation require no remote machine.

The fixtures are a fixed regression reference. `python -m script.diffsound.capture` can recapture them from the pinned author checkout when deliberately auditing source parity; it is not part of routine training or validation. End-to-end checks exercise the released tasks at bounded budgets. They do not establish convergence or agreement with every published table or figure.

See [model and usage](script/diffsound/README.md).

### Stable Modal Synthesis

```sh
script/ValidateStableModal --output build/stable-modal-training-check
```

The bundled captures establish independent CUDA agreement. Use `--golden path/to/accepted-local.pt.gz` for subsequent comparisons against an accepted local capture. Checks cover loss, states, gradients, Adam updates/moments, loss-only execution, streaming blocks, and explicit auxiliary-state preservation. The native target rebuild is also checked against the previous local implementation; numerical agreement does not establish a complete training reproduction.

Completed reproduction checks:

- Independent dense SAV solve, potential-gradient identity, a weight finite difference, and unforced undamped energy conservation.
- All 140 CUDA-generated recordings match the released normalized target audio within relative L2 `5.43e-14`; all linear reference recordings match exactly. Dataset fingerprints are retained in `build/stable-modal-cuda/results/dataset-comparison.json`.
- Released-checkpoint evaluation is complete for all 140 recordings. Normalized predictions match the published audio within relative L2 `1.39e-12`. All 12 Table 2 entries agree when the computed errors are truncated to two significant digits; the paper does not state its rounding convention. Full-precision metrics remain in the reports.
- CUDA Graph capture produces exactly equal states and audio to eager inference for the checked short trajectories, including a batch of four training strings.
- Native forward synthesis is checked against complete saved CUDA trajectories. Learned rendering shares the Accelerate training step and preserves states exactly across block sizes; the spectral generator uses Metal float-pair arithmetic.
- All five comparison figures have been generated from the released checkpoint and full test partition. The worst-case recording is 40 and the strongest-nonlinearity recording is 6, matching the author notebook selections. Styling differs from the paper.

Training ended after 409 completed epochs, with epoch 408 selected by validation loss. Final evaluation covers all 140 recordings and includes the five comparison figures. This is a shortened-training adaptation, not a reproduction of the full 2,000-epoch training experiment. Dataset regeneration and released-checkpoint evaluation are complete. Native training uses Accelerate float64 and supports full local dataset iteration, validation and checkpoint/resume; the full local training schedule has not been run.

#### Shortened training

Evaluation uses the checkpoint selected by validation loss at epoch 408 of the 409 completed epochs. The table reports the mean per-recording normalized squared error on the 60 test recordings; initial means the first 100 ms, and full means the complete three-second trajectory. The released column uses the full-precision results from our evaluation of the author checkpoint, avoiding ratios against rounded paper values.

| Test metric | Shortened run | Released checkpoint | Error ratio |
| --- | ---: | ---: | ---: |
| Modal displacement, initial | 0.00191286 | 0.000279280 | 6.85 |
| Waveform, initial | 0.00198884 | 0.000270907 | 7.34 |
| Modal displacement, full | 0.134677 | 0.0697109 | 1.93 |
| Waveform, full | 0.136552 | 0.0736534 | 1.85 |

All reported train, validation, and test means are higher than the released-checkpoint errors. This run demonstrates training and evaluation execution, but does not match the paper's trained quality. The independent released-checkpoint reproduction remains successful. These results do not isolate the effect of training duration from the other documented execution adaptations.

Artifacts are under `build/stable-modal-cuda/results/`: `training-stop.json` records the final decision, checkpoint epochs, optimizer steps, and hashes; `retrained-checkpoint/report.json` and `assessment.json` retain metrics and comparison details; `retrained-checkpoint/figures/` contains the five plots. `checkpoint_409.pt` and `checkpoint_best.pt` retain model and optimizer state. These generated files are ignored by Git.

See [model, training protocol and execution adaptations](script/stable_modal_synthesis/README.md).

## WaveBlender

```sh
script/FetchScenes CupPhone
script/ValidateGolden CupPhone
script/ValidateGolden --score CupPhone
```

Omit scene names for the complete suite. CUDA listener references live under
`gen/cuda/<scene>/<sample-rate>/`; fetch scene assets separately. Compare matching
simulation/sample rates. Scores are written to `gen/validation.json`.

Waveform, envelope and spectral agreement are scored separately. Native modal
stepping fixes an upstream batch-boundary bug retained in the Modal goldens, so
those references support envelope rather than waveform agreement. Other deliberate
differences include fresh-cell solves, overlapping source writes, bubble inverses
and scene timing. `ACOUSTIC_GLOBAL_LSTSQ=1` selects the upstream global fresh-cell
solve for diagnostics.

## SonicRadiation

```sh
build/RadiationTest --gate
script/ValidateRadiation
```

Analytic checks cover free-field behavior, boundaries, stability and motion;
scene regressions compare against `gen/radiation/` at a default 90 dB SNR gate.
Both grid-sampled and retarded-potential outputs are tested. There is no public
upstream executable providing a full trajectory reference.

Numerical departures include the near set, filtered feedback, diagonal boundary
solve, transport-grid remeshing and history-weight approximation. Scope is closed
rigid modal bodies, not every WaveBlender scene type.

## Room acoustics

```sh
build/RoomTest
script/ValidateRoom
build/RoomTest --implicit
```

Checks cover analytic modes, dispersion, boundary losses, energy and stability.
PFFDTD comparisons use matched grids, boundary data, sources and receiver clocks;
`gen/room/` captures provide separate native regressions.

Explicit frequency-dependent LRC walls and implicit real-admittance walls are
different models. Record optional mean projection when enabled. Processed WAVs
include differentiation, filtering, resampling and air absorption; use raw samples
for numerical comparisons. A truncated rigid-room tail is not physical damping.

## Porous room elements

```sh
script/ValidatePorous
script/ValidatePorous --cpu
script/ValidatePorous --demo
script/ValidatePorous --band  # extended sweep; requires matplotlib
```

Checks cover material fits, interfaces, box/OBJ classification, attenuation and
repeatability. Metal receiver error must be below `2e-4` relative to FP64.
Slab reflection/transmission is checked against an independent continuum model,
with normal/oblique incidence, backed/freestanding configurations and grid refinement.
`--band` extends the comparison to 20–4000 Hz. Reports and plots are under
`build/porous/`; they record the actual voxelized geometry, angles and error gates.

Fits require stable poles, positive relaxation weights and sampled passivity.
These tests do not prove stability for arbitrary materials or durations, or
reproduce the paper's full room benchmark. Nonzero initial porous fields require
consistent auxiliary states. The demo's 4 cm grid targets low frequencies; its
4 kHz material fit does not establish 4 kHz spatial accuracy.

## Immersed boundaries

```sh
build/ImmersedTest --gate
script/ValidateImmersed
```

Free-field and analytic sphere checks cover energy, boundary limits and stability.
Sphere/barrier regressions compare against `gen/immersed/` at a default 90 dB SNR
gate. `ImmersedTest --energy` and `--soak` provide longer runs.

The implementation uses an outer split PML and equal-area sphere sampling. Scope
is static exterior surfaces and barriers, excluding moving surfaces and conservative
enclosure walls. Surface matrices require quadratic memory in patch count.

## Curved tetrahedral DG

```sh
build/DgReference prepare build/dg-inputs
build/DgReference check build/dg-reference-checks
build/DgReference validate build/dg-metal build/dg-inputs
```

An independent FP64 implementation and analytic rigid-cylinder solution check
geometry, mass inversion, operators and propagation. Gates include analytic field
error below `1e-3`, FP64 state/receiver/operator errors below `2e-5`, constant-pressure
preservation, energy/mean bounds and repeatable outputs. Checks also cover timestep
refinement, alternate quadrature, working-set sizes and invalid inputs.

The stock degree-8 cylinder is outside validated FP32 scope; fixture checks do not
establish a timestep bound for arbitrary meshes. The
[upstream CUDA comparison](src/Dg/README.md#upstream-cuda-comparison) uses matched
inputs but different spatial formulations, so curved-mesh waveforms need not match
exactly. See [formats and reference commands](src/Dg/README.md).

## Nonlinear plates

```sh
build/PlateTest --gate
script/ValidatePlate --output build/plate-validation/dense
script/ValidatePlate --fft --output build/plate-validation/fft
```

Initialize `external/fa2026` for comparisons with the pinned, unmodified Python
reference. The wrapper manages an isolated environment and records hashes.
`--score` checks saved artifacts; `--reuse-reference` reuses verified captures.
Checks cover nonlinear gradients, transforms, energy/work, regulator norms,
initialization, repeated strikes, physical scaling and both Metal backends.
[ValidatePlate](script/ValidatePlate) defines per-case tolerances, including wider
limits for the roundoff-sensitive symmetric regulated case.

### High-precision Metal accuracy

```sh
script/PlatePrecisionRender --output build/plate-reference/crash
script/ValidatePlateAccuracy --reference build/plate-reference/crash --output build/plate-accuracy/crash
```

The three-second unregulated small crash is checked against a converged,
independent high-precision discrete reference. Gates require `1e-8` relative
agreement for waveforms, modal histories, final state, diagnostics and STFT
magnitude, plus energy/work balance within `2e-9` of peak energy. This establishes
discrete convergence, not upstream FP64 waveform parity or continuum accuracy.

Use `PlatePrecisionRender --case NAME` for `config/plate/` fixtures or `--config PATH`
for supported scaled configurations. Give each reference its own output directory.
`--steps` provides a bounded smoke check; `ValidatePlateAccuracy --score` checks
saved results. Two nine-second repeated-strike cases lack converged full-duration
references, and the 400:1 aspect-ratio case exceeds its CPU potential-error gate.

## Nonlinear stiff strings

```sh
build/StringTest --metal
brew install octave
python3 -m pip install numpy soundfile
git submodule update --init external/nLinStringsConv-SAV
script/ValidateString
script/ValidateString --long --output build/string-validation-long
```

Native checks cover gradients, conservation/damping, analytic modes, timestep
refinement and CPU/Metal agreement. Octave comparisons cover all seven variants
and both integrators, including grids, clocks, displacement, states and available
diagnostics. Regular captures last 2 ms; `--long` extends them to 60 ms, with
small-shift smoke cases capped at 0.5 ms. `--cpu-only` skips Metal;
`--reuse-reference` verifies cached inputs and hashes.

Geometric Form A is sensitive to timestep, grid and potential shift; `--stress --long`
exercises cases that can fail waveform agreement. Form B's longitudinal motion
requires a suitable timestep. Upstream stale/unfilled energy columns are excluded
from parity checks, and its unsplit damping update lacks the split scheme's energy
balance guarantee. The native linear limit is checked analytically because the
upstream linear branches are inconsistent.

### Paper experiments

```sh
python3 -m pip install numpy matplotlib
python3 script/string/PaperOde.py
python3 script/string/PaperStrings.py --workers 4
```

Sweeps reconstruct Figures 1–7 and compare Tables 2–3, writing results, figures and
provenance under `build/string-paper/`. `--plot-only` reuses results; ODE `--smoke`
and lower string `--max-exponent`/`--reference-exponent` values bound the work.
Some geometric cases terminate numerically, and table values are not reproduced
exactly; reports retain failures and interpretation choices.

## Coupled plates and air

```sh
git submodule update --init external/ness
script/ValidateCoupled
script/ValidateCoupled --demo
```

Metal stepping is compared with pinned NESS FP64 and a separate sparse direct solve
for explicit rectangular edges. Cases cover nonlinear/linear panels, sympathetic
motion, air loading, repeated strikes, drum heads and mixed boundaries. Receiver
and complete final-state relative L2 errors must be below `1e-4`.

`CoupledBoundaryTest` checks stiffness symmetry, free-plate rigid modes, nonlinear
energy, interpolation work and frequency refinement against analytic and
[Leissa](https://ntrs.nasa.gov/citations/19700009156) references. CPU/Metal comparisons
share prepared coefficients; these independent checks assess the boundary operators.
Native pressure includes the `rho/2` factor omitted by NESS's output convention.
Generated NESS copies receive bounds/initialization fixes; the submodule is unchanged.
These checks do not establish convergence of every coupled nonlinear trajectory.

## Acoustic Reliefs

See [sanity and upstream comparison commands](script/reliefs/README.md). Tests compare
Metal pressure/height gradients against dense FP64 and matched mesh/target states
against the authors' implementation. Acoustic and appearance objectives are scored
separately; optimization trajectories may differ under rounding.

## Viscothermal tubes

```sh
git submodule update --init external/openwind
python3 -m venv --system-site-packages build/tube-reference/venv
build/tube-reference/venv/bin/python -m pip install numpy scipy matplotlib h5py soundfile
build/tube-reference/venv/bin/python script/ValidateTube --demo
```

Pinned OpenWind v0.12.4 provides independently assembled operators and propagation
with matched materials, geometry, boundaries and excitation. Preparation, receiver
pressures, final acoustic/memory states and energy must agree within `1e-7` relative
L2. Cases cover taper, polynomial order, loss approximations, endpoint conditions,
silence, conservation and passive decay; invalid inputs must fail before stepping.

A cylinder refinement check compares rendered impedance against the independent
Zwikker–Kosten solution over 100–3000 Hz, requiring improvement and final error
below 2%. `build/tube-validation/` contains traces, WAVs, reports and hashes;
`--output` and `--executable` select alternatives. `--demo` adds the five-second
README example. Exact paper trumpet plots and perceptual equivalence are not claimed.

## Coupled elastic–acoustic waves

Build `ElasticTest`, `ElasticReference` and `ElasticSolve` for the native checks:

```sh
build/ElasticTest build/elastic-validation.json
```

Checks compare element-local GPU propagation with assembled FP64 and direct CPU
quadrature. Coverage includes curved heterogeneous anisotropic media, both mass
operators, interface energy, receiver/diagnostic agreement, source work, Scholte
waves and high-impedance transmission. Small-model spectra check the operator CFL
bound and RK stability; mesh/order and timestep refinement are assessed separately.

Independent references require initialized `external/wadg-elastic`, `external/e3dss`
and `external/seissol` submodules, plus NumPy, Gmsh, h5py and Octave:

```sh
python3 script/elastic/author_reference.py
python3 script/elastic/sphere_reference.py
# SeisSol also requires Homebrew LLVM, Eigen, OpenMPI and Ninja.
python3 script/elastic/build_seissol.py
python3 script/elastic/seissol_reference.py --scenario scholte
python3 script/elastic/seissol_reference.py --scenario snell --cells 4
```

The author harness executes the 2D MATLAB example with a recorded boundary-clock
correction. e3Dss supplies an independent harmonic sphere solution; SeisSol supplies
matched Scholte/Snell trajectories. Both native and SeisSol results are also scored
against analytic fields. Reports and hashes live under `build/elastic-reference/`.

`build/ElasticTest --viscoelastic` checks the Zener complex modulus against the
pinned SeisSol attenuation fit, analytical stress relaxation, P/S plane-wave
convergence, the lossless limit, and CPU/Metal agreement for curved, heterogeneous,
anisotropic mixed fluid/solid meshes. It checks full and weight-adjusted mass,
multiple mechanisms, energy balance with separate material dissipation, and the
augmented operator CFL spectrum. These checks are also part of `ElasticTest`.
The full geological benchmark suite from Shukla et al. has not been reproduced.

The local absorbing boundary reflects oblique waves,
and demo meshes have no guaranteed frequency bandwidth. These checks do not qualify
measured specimens or provide the paper's photoacoustic inversion workflow.

## Reproducing results

Match inputs, discretization, clocks and output precision; retain report hashes.
For timing, record hardware, warmup, repeated runs, competing load and excluded
setup/output work. Runtime Metal shaders come from the source checkout recorded
at build time: separate binaries from one checkout share that shader source.

## NumCalc scattering and radiation

`script/ValidateNumCalc --head` compares native CPU and compensated Metal pressure/velocity fields with the pinned, unmodified author executable. Cases cover both source types, boundary pressure/velocity/admittance, frequency curves, direct/CGS solves, ILU and scaling, dense/single-level/multilevel FMM, and the author's sample head. Independent sphere solutions, mesh refinement, fictitious-resonance frequencies and the paper's terminated-duct example check physical accuracy. The native residual is recomputed in FP64 against the unpreconditioned operator.

The numerical core is shared with upstream, so those comparisons establish execution-backend agreement; the analytic checks supply independent evidence. The sample head is smaller than the published head-and-torso example. Full published figures and timing tables have not been reproduced. See [scope, conventions and rendering checks](src/NumCalc/README.md).

## Thermoviscous boundary elements

`script/ValidateThermoviscous` compares native quadratic boundary operators and all recovered fields with the pinned Julia implementation's direct coupled solve at 100 and 1000 Hz. The harness explicitly corrects upstream free-term and interior-normal conventions. Independent mesh-refinement checks cover the paper's translating sphere and analytical radial interior/exterior thermoviscous solutions, plus complex excitation and silence. A disconnected narrow-gap case checks quadrature convergence and agreement with a direct coupled solve.

The extension checks cover all four solve formulations, FMM/H-matrix agreement on both fluid sides, compressed operator probes, and mesh refinement for linear/quadratic quads and discontinuous triangle/quad fields. A delayed band-limited pulse checks Fourier sign and linear convolution. `--analytic-only` omits Julia; `--extensions-only` runs just the added paths. Reports go to `build/thermoviscous-validation/`.

`script/BenchmarkThermoviscous` covers all seven sphere meshes at 100/1000 Hz, recovering the deleted 2,794-element mesh from upstream history. Use `--methods iterative direct1 direct4 direct10 --repetitions 20` for the paper's comparison protocol; direct cases over the explicit memory budget are recorded as skipped. Local timings use different hardware and report their timing scope.

Broadband rendering checks impulse guard energy and reports frequency spacing, but users must refine the FFT grid to resolve narrow resonances and long cavity decay. Neither a small algebraic residual nor a passing guard qualifies arbitrary geometry or broadband accuracy. See [solver conventions and limits](src/Thermoviscous/README.md).
