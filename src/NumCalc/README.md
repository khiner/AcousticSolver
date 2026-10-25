# Acoustic scattering and radiation

Port of Kreuzer, Pollack, Brinkmann and Majdak, [*NumCalc: An open-source BEM code for solving acoustic scattering problems*](https://doi.org/10.1016/j.enganabound.2024.01.008), Engineering Analysis with Boundary Elements 161, 2024. The author implementation is pinned at Mesh2HRTF v1.3.0 (`e45d0436a6fbeca3db13828cbae23ca109225be3`) in `external/mesh2hrtf` under EUPL-1.2.

This solves linear, frequency-domain acoustics in a homogeneous fluid, including exterior scattering/radiation and bounded cavities. Constant triangular or planar quadrilateral elements support prescribed pressure, normal velocity, and frequency-dependent surface admittance. Sources may be point sources, plane waves, or prescribed boundary excitation. Exterior solves use Burton–Miller stabilization to remove fictitious resonances. Interior physical resonances remain.

## Run

```sh
git submodule update --init external/mesh2hrtf
cmake --build build --target NumCalcSolve NumCalcReference
script/PrepareNumCalc build/numcalc/sphere.json
script/RunNumCalc build/numcalc/sphere.json --output build/numcalc/sphere
script/RunNumCalc build/numcalc/sphere.json --output build/numcalc/sphere-metal --backend metal
script/PrepareNumCalc build/numcalc/enclosure.json --scene enclosure
script/PrepareNumCalc build/numcalc/duct.json --scene duct
```

The output directory contains the prepared NumCalc project, execution log, provenance hashes and `result.json`. Each frequency has boundary pressure in Pa, outward-normal boundary velocity in m/s, receiver pressure, and three Cartesian receiver velocity components. Complex numbers are `[real, imaginary]`; receiver fields include the incident wave. `native.json` records the actual dense/FMM method, algebraic residual, operator storage and assembly/solve timing.

`script/RunNumCalc --reference` executes the unmodified author solver on the same prepared input. Existing NumCalc projects can run directly with `build/NumCalcSolve PROJECT_DIR --backend cpu`; the author command-line frequency selectors, iteration limit and memory estimator are retained.

## Geometry and boundary data

The editable JSON contains `vertices` in metres and `elements`, a list of zero-based triangles or quads. Surfaces must be closed and consistently wound outward from the bounded region: the obstacle for exterior problems, the fluid for interior problems. Quads must be planar. Multiple disconnected closed objects are supported. Receivers must be in the fluid, off the boundary and away from point-source singularities.

```sh
script/PrepareNumCalc build/numcalc/imported.json --mesh surface.msh --scale .001
```

Mesh import reuses the Gmsh/COMSOL surface reader; Gmsh files require the Python `gmsh` package. Only linear triangles and quads are accepted. Set receiver positions and excitation in the generated JSON for your geometry.

| Key | Meaning / default |
| --- | --- |
| `frequency` / `frequencies` | Positive Hz; default 500; sweeps strictly increasing |
| `sound_speed`, `density` | m/s and kg/m³; defaults 343 and 1.21 |
| `exterior` | `true` for unbounded exterior, `false` for a cavity |
| `boundary` | Default for every element; `{"type":"velocity","value":0}` |
| `patches` | Boundary objects with an `elements` index list; override the default |
| `sources` | List of plane or point sources; empty by default |
| `receivers` | Fluid coordinates in metres |
| `method` | `dense`, `slfmm`, or `mlfmm` (default) |
| `solver` | `cgs` (default) or `direct`; direct requires dense and CPU |
| `preconditioner` | `scaling` (default), `ilu`, or `none` |
| `expansion_terms` | Minimum FMM expansion length, 5–10; default 8 |
| `cluster_edge` | Root cluster box length in metres; zero selects the author's rule |
| `max_iterations` | CGS iteration limit; default 500 |
| `symmetry` | Optional `planes` (three author codes 0–4) and `origin` in metres |

Boundary types are `velocity` (m/s), `pressure` (Pa), `admittance` (normalized by multiplying physical admittance by `density * sound_speed`), and `impedance` (normalized by dividing physical impedance by that factor). A rigid surface has zero velocity; a pressure-release surface has zero pressure. An admittance boundary can also contain a prescribed `velocity`.

Values accept a real number, `[real, imaginary]`, or a curve `{"frequencies":[100,1000],"values":[[0.1,0.02],[0.3,0.05]]}`. Curves use linear interpolation and must cover the solved frequencies. Frequency curves are supported for admittance and source amplitudes. Pressure/velocity curves are evaluated by the frontend for individual frequency solves, including audio sweeps. Use admittance curves for frequency-dependent impedance.

Plane sources use `{"type":"plane","direction":[1,0,0],"amplitude":1}`; amplitude is Pa at the coordinate origin. Point sources use `{"type":"point","position":[1,0,0],"amplitude":1}`; amplitude has units Pa·m, with incident pressure `amplitude * exp(i*k*r)/(4*pi*r)`. Both use `exp(-i*omega*t)` time dependence.

## Numerics and performance

The submodule supplies geometry integration, singular and hypersingular quadrature, boundary assembly, CGS/ILU, FMM clustering and field recovery. Checked build-directory adaptations connect native operator applications and Accelerate LU, retain full output precision, and normalize the CGS stopping criterion by the RHS norm. Upstream instead normalizes by the first iteration residual, which can allow a loose physical-system residual at low frequencies. Every native solution also passes a separately recomputed FP64 residual check against the unpreconditioned operator (`1e-7` relative limit).

CPU uses FP64 and Accelerate dense algebra, with parallel sparse FMM operations. Metal packs each real component as a high/low float pair, uses compensated products and reductions, and keeps CGS and preconditioning in FP64 on the CPU. This is higher precision than FP32, with float's exponent range; it is not hardware FP64. Assembly and receiver evaluation remain on the CPU. GPU buffers duplicate the retained operator. Small cases favor CPU, so CPU is the default.

Native adaptations also accept two boundary records per element and recognize CGS convergence on the final allowed iteration. If dense CGS fails, the author's direct fallback uses Accelerate on the restored original matrix and RHS, including when Metal was requested. The frequency report identifies this as `solver: direct_fallback` and `direct_backend: accelerate`. Failed FMM solves must still pass the residual check; they do not fall back to dense LU.

```sh
script/RunNumCalc build/numcalc/sphere.json --output build/numcalc/bench --benchmark 50
```

This measures warm CPU/Metal applications of the same assembled operator and compares their results. Measurements include vector transfer and synchronization, exclude setup, and are separate from end-to-end solve timing. `operator_bytes` counts host coefficient/index storage, excluding geometry, scratch space, residual-check copies, direct factorization and GPU copies.

The author's frequency/geometry-dependent selection can reduce a requested multilevel solve to single-level or dense. Reports identify the actual method. Default multipole truncation follows the paper; the later upstream experimental `-adapt_fmmlength` option is not enabled.

This is a constant-element approximation. Refine geometry, frequency sampling and the FMM parameters for each application. Narrow gaps, irregular elements, high frequencies and cavity resonances can demand substantially finer discretization. A small algebraic residual does not establish discretization accuracy. The model has no mean flow, nonlinear propagation, viscothermal boundary layers, moving geometry or two-way structural coupling; prescribed wall velocity drives radiation.

## Audio and validation

```sh
script/RenderNumCalc build/numcalc/sphere.json --output build/numcalc/audio
script/RenderNumCalc build/numcalc/sphere.json --input excitation.wav --fft-size 2048
script/ValidateNumCalc --head
```

Rendering solves and caches positive-frequency bins, conjugates the transfer function for NumPy's inverse FFT, and performs linear convolution with mono excitation. The default input is a pulse in a two-second buffer. Input samples multiply all configured sources and prescribed pressure/velocity; admittance stays fixed. Output includes the convolution tail. `pressure.npy` retains physical Pa values; `render.wav` is normalized, with playback gain recorded in `report.json`.

Defaults are 8 kHz sampling and a 512-sample impulse period. The tapered band limit is the smaller of 3.6 kHz and eight edges-per-wavelength resolution (`c / (8 * largest_edge)`); `--max-frequency` overrides it. This is a nominal mesh guideline, not an accuracy guarantee. Static/DC pressure is excluded. A quarter-period time-origin shift accommodates receivers upstream of a plane wave's origin. The renderer rejects excessive energy in the impulse guard intervals; increase `--fft-size` and compare successive responses to resolve long decay and narrow resonances. Guard energy alone is not a convergence proof.

Validation compares pressure and velocity fields with the unchanged author executable for dense, single-level and actual multilevel solves, source/boundary variants and both native backends. Independent checks cover sphere solutions and refinement, fictitious-resonance frequencies, silence, and the paper's 3.4 m duct at 240/480 Hz with 0.05/0.025 m panels. `--head` adds the author's 17,432-element sample head. That sample is not the much larger published head-and-torso mesh; the complete published figure/timing study has not been reproduced. Reports and fixture hashes are written under `build/numcalc-validation/`.
