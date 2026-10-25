# [Coupled elastic–acoustic waves](https://doi.org/10.1016/j.jcp.2020.109632)

`ElasticSolve` implements the linear stress–velocity and pressure–velocity DG
formulation of Guo, Acosta and Chan (JCP 418, 2020). Tagged tetrahedra contain
fluids or elastic solids, with conforming curved interfaces, heterogeneous
coefficients and symmetric positive-definite anisotropic stiffness. The interface
transmits normal velocity and traction, permits fluid–solid tangential slip, and
sets fluid-side shear traction to zero. Solid–solid interfaces are bonded.

The offline solver keeps propagation, receivers and energy diagnostics on the GPU,
using paired-float arithmetic and FP64 preparation. GPU rendering skips assembly
of the global evolution matrix: it applies element-local volume operators,
preintegrated face fluxes and quadrature-factorized weight-adjusted inverse mass.
Preparation still forms local matrices for energy diagnostics and the CFL bound.
`--cpu` selects the assembled Accelerate FP64 reference with the same fixed timestep. Both use five-stage fourth-order
Runge–Kutta integration and either full physical mass inversion or matrix
weight-adjusted mass inversion. Solution degrees 1–6 and geometry degrees
1–4 are supported. This is small-deformation, lossless acoustics and elasticity;
it does not model contact, plasticity, viscosity, material attenuation or moving
interfaces. Numerical dissipation is reported separately from physical energy.

## Run

Build with the [repository instructions](../../README.md#build). Mesh preparation
requires Python, NumPy and Gmsh (`python3 -m pip install numpy gmsh`). Native solving
does not require Python or any reference solver.
GPU rendering requires macOS 26 or newer and a Metal 4-capable Apple GPU.

```sh
python3 script/ElasticDemo sphere build/elastic-demo/sphere
build/ElasticSolve build/elastic-demo/sphere/config.json --prepare
build/ElasticSolve build/elastic-demo/sphere/config.json --output build/elastic-demo/sphere/render

python3 script/ElasticDemo container build/elastic-demo/container --spacing 0.3
build/ElasticSolve build/elastic-demo/container/config.json --prepare
```

The examples prepare a soft elastic sphere in air and an acrylic spherical shell
containing water. They request one second of output. `--seconds N` overrides the
duration. `--prepare` reports the unknown count, preparation time and CFL timestep
without rendering. Fine solid elements and material interfaces can require many
steps per audio sample; these simulations can take substantially longer than the audio.
Example meshes are starting resolutions, not a prescribed accuracy level. Refine
the mesh/order and compare receiver traces over the frequency band of interest.

## Mesh and configuration

```sh
python3 script/PrepareElastic input.msh build/elastic/mesh.json
build/ElasticSolve config.json --output build/elastic/render
```

Assign exactly one physical volume tag per Gmsh region and, optionally, one physical
surface tag per exterior boundary. Leave internal interfaces untagged as boundaries.
Use conforming tetrahedra of uniform geometry degree; the converter preserves
high-order geometry nodes. Inverted, degenerate and nonconforming elements are
rejected. Surface meshes alone are insufficient.

All inputs use SI units. Paths are relative to the configuration file. For example:

```json
{
  "mesh": "mesh.json",
  "degree": 2,
  "mass": "weight_adjusted",
  "seconds": 1.0,
  "sample_rate": 48000,
  "default_boundary": "absorbing",
  "materials": {
    "1": {"type": "fluid", "density": 1.204, "sound_speed": 343.0},
    "2": {"type": "solid", "density": 1180.0, "young_modulus": 3e9, "poisson_ratio": 0.35}
  },
  "boundaries": {"10": "rigid", "20": "free"},
  "sources": [
    {"type": "force", "position": [0.1, 0.02, 0.01], "region": 2,
     "direction": [1, 0, 0], "amplitude": 1.0, "frequency": 250.0, "time": 0.006}
  ],
  "receivers": [
    {"position": [0.6, 0.11, 0.03], "region": 1, "field": "pressure"}
  ]
}
```

Material keys are physical volume tags; boundary keys are physical surface tags.
A fluid accepts `bulk_modulus` (Pa) in place of `sound_speed` (m/s). A solid accepts
a 6×6 `stiffness` matrix (Pa) in place of Young's modulus and Poisson's ratio.
Voigt order is `xx, yy, zz, yz, xz, xy`, with engineering shear strain. Density,
moduli and stiffness entries can be numbers or spatial fields:

```json
{"value": 1000, "gradient": [10, 0, 0],
 "sinusoid": {"amplitude": 20, "wavevector": [2, 0, 0], "phase": 0}}
```

The field is `value + gradient·x + amplitude*sin(wavevector·x + phase)`.
Coefficients must remain physically admissible throughout each element, including
between quadrature points. Sub-cell variation needs sufficient quadrature and mesh
resolution; the solver samples admissibility at its integration points. `quadrature_order`
can increase integration accuracy above its default `degree + geometry_degree + 1`
(up to 20); check convergence for rapid coefficient variation.

`mass` is `full` or `weight_adjusted` (default). `penalty` defaults to 1; zero selects
the central interface flux. Integration uses fixed-step, five-stage fourth-order
Runge–Kutta on both backends, with no embedded error estimate or adaptive retries.
The timestep is chosen before rendering from the operator CFL bound below. A positive
`dt` overrides it; choosing a larger value requires stability and timestep-refinement
checks for the specific model.
The requested step is rounded down to an integer number of substeps per audio sample
and remains fixed for the entire render. Refine `dt`, mesh size and polynomial order
to check temporal and spatial convergence separately.

The 3D CFL calculation follows the symmetric/skew spectral enclosure in
[Chan et al. (2016), §3.1](https://doi.org/10.1016/j.jcp.2016.04.003).
For the assembled system `M q′ = L q`, factor the discrete energy mass as `M = UᵀU`
and form element blocks of `A = U⁻ᵀ L U⁻¹`. With `S = (A+Aᵀ)/2` and
`K = (A−Aᵀ)/2`, compute

```
a = max_i sum_j ||S_ij||₂
b = max_i sum_j ||K_ij||₂
dt_CFL = 3 / sqrt(a² + b²)
```

The block norms are largest singular values, evaluated with Accelerate.
[Block norm bounds](https://msp.org/pjm/1962/12-4/pjm-v12-n4-p08-s.pdf)
give `−a ≤ Re(λ) ≤ 0` and `|Im(λ)| ≤ b` for the energy-stable discretization.
This evaluates the published spectral-bound construction on our actual 3D matrices,
including curved geometry, either mass formulation, anisotropy, interfaces and
boundary penalties; it does not reuse the author's 2D timestep guess or the
acoustic-only trace-constant estimate. The retained
[Carpenter–Kennedy RK method](https://ntrs.nasa.gov/citations/19940028444) has
stability polynomial `R(z) = 1+z+z²/2+z³/6+z⁴/24+z⁵/200`;
the left half-disk `|z| ≤ 3` lies inside `|R(z)| ≤ 1`.
This is a linear spectral stability restriction, not a temporal error tolerance
or a guarantee of energy decrease at every RK step. Timestep refinement remains
necessary for accuracy.

Sources are Ricker pulses `amplitude*(1-2*a*a)*exp(-a*a)`, where
`a = pi*frequency*(t-time)`. `force` applies the specified direction times an
amplitude in newtons; direction is not normalized. `volume_flow` is a fluid monopole
in m³/s. They are DG point loads, so source-near-field values depend on resolution.
Optional `region` disambiguates source/receiver placement at interfaces. Receivers
accept `pressure` in fluids, `vx/vy/vz` in either medium, and
`sxx/syy/szz/syz/sxz/sxy` in solids.

## Boundaries

Fluid boundaries support `rigid`, `pressure_release` and `absorbing`; solids support
`fixed`, `free` and `absorbing`. `prescribed` is a CPU-only native C++ callback boundary used
by the reference harness, not a standalone JSON incident-field specification.

`absorbing` is a local characteristic boundary, not a PML or an exact free-space
boundary. For a plane acoustic wave incident at angle θ to its normal, its
continuum pressure reflection coefficient is `(cos(θ)-1)/(cos(θ)+1)`: zero at normal
incidence, about −0.172 at 45°. Refine and enlarge the exterior domain to measure
boundary contamination; an accurate spatial discretization does not remove this
reflection. Cavity and interface convergence tests do not qualify arbitrary
exterior radiation problems.

## Output

For output prefix `render`, the solver writes:

- `render.bin`: sample-major, interleaved FP64 receiver values in physical units.
- `render.wav`: 24-bit WAV with a common peak-normalization gain recorded in JSON.
- `render.energy.bin`: sample-major FP64 columns for discrete energy, physical
  energy, source power, boundary power and interior interface dissipation.
- `render.state.bin`: terminal FP64 Bernstein coefficients, ordered by mesh element,
  then field, then basis coefficient.
- `render.json`: configuration, step count, fixed timestep (`dt`),
  CFL limit (`cfl_dt`) and real/imaginary spectral bounds (`cfl_real_bound`,
  `cfl_imaginary_bound`, in s⁻¹), model/backend preparation times, render time, GPU command time (`gpu_seconds`,
  zero on CPU), WAV gain and column names.

Render timing includes stepping, diagnostics and output download;
it excludes model/backend preparation and file writing.

Sample zero is at `1/sample_rate`. Raw files use native little-endian doubles.
The fluid state is `[vx,vy,vz,-pressure]`; solid state is
`[vx,vy,vz,sxx,syy,szz,syz,sxz,sxy]`. Receiver pressure has the physical compression
sign. Channels with different physical units share the WAV gain; use raw values for
quantitative comparisons.

Discrete energy obeys `dE/dt = source_power + boundary_power - interface_dissipation`
up to integration/roundoff error. Boundary power includes boundary numerical loss
and any prescribed boundary input. With weight-adjusted mass, physical and discrete
energy are distinct; neither interface penalties nor absorbing-boundary loss are
material damping.

See [validation and reference commands](../../VALIDATION.md#coupled-elasticacoustic-waves).
