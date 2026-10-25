# Thermoviscous boundary elements

Port of Preuss et al., [*Revising the Boundary Element Method for Thermoviscous Acoustics: An Iterative Approach via Schur Complement*](https://doi.org/10.1142/S2591728523500159), JTCA 31(4), 2023. The [MIT-licensed author implementation](https://github.com/mipals/BoundaryIntegralEquations.jl) is pinned at `938ee0630a7a6516c15dea33998472bdbdcbbfc0` in `external/thermoviscous-bem`.

This solves linear, harmonic acoustics in a homogeneous viscous, heat-conducting fluid with a prescribed wall velocity and an isothermal, no-slip boundary. It includes acoustic and thermal pressure modes and the divergence-free viscous velocity mode. Useful applications include small cavities, narrow gaps, microphone geometries and sound radiation from small vibrating surfaces. There is no mean flow, turbulence, moving mesh, two-way elastic coupling or finite-amplitude acoustics.

## Run

```sh
cmake --build build --target ThermoviscousSolve
script/PrepareThermoviscous build/thermoviscous/sphere.json --radius .01 --frequency 1000
build/ThermoviscousSolve build/thermoviscous/sphere.json build/thermoviscous/result.json
script/ThermoviscousDemo                         # two-second exterior listening sample
script/ThermoviscousDemo --interior --output build/thermoviscous-cavity
```

The demo writes normalized `steady.wav`, physical pressure and timing in `result.json`, and the playback gain in `audio.json`. This is a steady sinusoidal response with a listening fade, not a simulated attack or decay. Use `RenderThermoviscous` below for pulse or WAV excitation. Julia is needed only for reference validation; Python tools need NumPy, SciPy and SoundFile. Initialize submodules and install Homebrew `gcc` (gfortran) before configuring CMake; the native FMM library is built with the solver.

## Geometry and excitation

```sh
script/PrepareThermoviscous build/thermoviscous/model.json --mesh surface.msh --frequency 1000
```

The importer accepts linear/quadratic Gmsh triangles and quadrilaterals (types 2, 9, 3, 16 and 10; Python `gmsh` required) and COMSOL `.mphtxt` surface exports. It writes editable JSON. Coordinates use metres. Meshes must be closed and consistently wound; quadratic neighbors share edge-midpoint nodes. Triangles have 3 or 6 nodes; quads have 4, 8 or 9. JSON ordering is counterclockwise corners, then edge midpoints in the same order, then the center for nine-node quads. Orient elements outward from the bounded region: the obstacle for an exterior problem, or the fluid for an interior problem. Refine geometry and quadrature to resolve narrow gaps and curved walls; boundary elements avoid volume meshing of the thin loss layers but do not remove surface discretization error.

The JSON contains:

| Key | Meaning / default |
| --- | --- |
| `vertices`, `triangles`, `quadrilaterals` | Geometry coordinates and zero-based connectivity |
| `physics_order` | `geometry` (default), `linear`, or `quadratic`; continuous fields cannot exceed geometry order |
| `discontinuous`, `physics_inset` | Element-local collocation fields; defaults `false`, `0.1`; inset must be between 0 and 1/3 |
| `frequency` | Positive frequency in Hz; default 100 |
| `exterior` | Exterior radiation (`true`) or bounded fluid (`false`) |
| `velocity` | Uniform Cartesian wall velocity in m/s; default `[0,0,0.01]` |
| `nodal_velocity` | Optional velocity vector per collocation point; overrides `velocity` |
| `receivers` | Optional fluid points strictly off the boundary |
| `material` | Optional SI parameters listed below; default air at 20°C, 101325 Pa, 50% RH |
| `acoustic_backend` | `dense` (default), `fmm`, or `hmatrix` |
| `compression_tolerance` | FMM/partial-ACA relative tolerance; default `1e-10`; independent of quadrature accuracy |
| `fast_quadrature`, `leaf_size` | Far triangle quadrature points (3, 7 or 12; default 12) and H-matrix cluster leaf size (default 64) |
| `method` | `iterative` (default), `direct1`, `direct4`, or `direct10`; direct methods require dense acoustic operators |
| `direct_memory_gib` | Explicit allocation estimate limit for direct comparisons; default 2 GiB |
| `quadrature` | Acoustic Gauss order per direction; default 6; also sets far quad quadrature |
| `quadrature_tolerance` | Minimum relative subdivision size for singular/loss-layer quadrature; default `1e-6` |
| `loss_neighborhood` | `distance` (default) retains nearby nonadjacent surfaces; `incident` reproduces the author's one-ring truncation |
| `loss_decay` | Distance cutoff in the larger boundary-layer thickness; default 25 |
| `tolerance`, `inner_tolerance` | Outer and nested GMRES relative tolerances; defaults `1e-7`, `1e-10` |
| `restart`, `max_iterations` | Fixed GMRES restart length and iteration limit per solve; defaults 120 and 1000 |

Complex velocity components accept `[real, imaginary]`; real numbers are also accepted. Material keys are `density` (kg/m³), `sound_speed` (m/s), `viscosity` and `bulk_viscosity` (Pa·s), `conductivity` (W/(m·K)), and `heat_capacity_p` and `heat_capacity_v` (J/(kg·K)). The defaults use the author's unrounded air properties. All quantities follow the paper's `exp(-i omega t)` convention.

`collocation_points` records the field ordering. Continuous fields use ascending retained geometry-node indices; discontinuous fields use element order then local node order. `nodal_velocity` follows that ordering.

Output pressure and derivatives use `[real, imaginary]`. `acoustic_pressure` and `thermal_pressure` are nodal Pa values; their sum is total pressure. `receiver_pressure` contains their sum at the requested points. `viscous_velocity` and `viscous_derivative` store all x components, then y, then z. Normal derivatives point **into the fluid**, so they are the negative of the paper's outward-domain normal derivatives. No-slip velocity and isothermal temperature are imposed through the coupled system.

## Numerics and limits

Continuous and discontinuous linear/quadratic collocation elements, surface derivatives, graded singular quadrature and nested restarted GMRES implement the one-pressure-unknown-per-point Schur system. All ten original fields are recovered. Constant fields are excluded because they cannot resolve the required surface gradients. The native direct 1-, 4- and 10-field systems provide the paper's comparison formulations; large direct solves are explicitly memory-budgeted.

All backends use FP64 on the CPU, with Accelerate for dense algebra. Thermal and viscous operators are sparse. Dense acoustic matrices require approximately `32 * points²` bytes. `fmm` uses the author's quadrature-source acceleration approach with [FMM3D](https://github.com/flatironinstitute/FMM3D); `hmatrix` uses geometric clustering and partial adaptive cross approximation from [HMatrices.jl](https://github.com/IntegralEquations/HMatrices.jl). Both retain local singular/near-field corrections. These extensions appear in the broader author code; the paper itself presents FMM as future work.

FMM avoids dense acoustic storage. H-matrix efficiency depends on frequency, geometry and tolerance; its quadrature-source matrices can be larger than dense nodal matrices on small problems or at tight tolerances. `acoustic_bytes` estimates retained operator storage, excluding temporary FMM workspaces. Setup still scans point/element pairs for nearby interactions. Refine `fast_quadrature` as well as the mesh: compression tolerance does not bound discretization error.

Two upstream conventions are corrected explicitly in the Julia validation harness: the free term follows the constant interior Laplace identity, and interior normal gradients reverse with the fluid side. The submodule remains unmodified. Spatial cutoff retains interactions between nearby disconnected walls, using the author's subdivision quadrature for nonincident loss integrals. Increase surface resolution and quadrature, tighten the subdivision cutoff and increase `loss_decay` to establish convergence for a new geometry. `incident` truncation is suitable only when omitted surfaces are well outside both loss layers.

The direct boundary integral formulation retains the usual exterior fictitious-resonance limitation. This port does not add Burton–Miller stabilization. A small algebraic residual alone does not establish physical accuracy near those frequencies or on an unresolved mesh.

```sh
script/ValidateThermoviscous                 # Julia operator/full-system oracle and analytic refinement
script/ValidateThermoviscous --analytic-only
```

The validator creates an isolated Julia environment under its output directory. Use `--julia /path/to/julia` if Julia is not on PATH. Its translating-sphere oracle is the paper's Appendix C approximation, which neglects thermal pressure; radial interior/exterior checks include both acoustic and thermal modes. A two-sphere narrow-gap check compares quadrature orders and the Schur solution against a dense coupled solve. Reports and generated fixtures live under `build/thermoviscous-validation/`.

## Broadband audio

```sh
script/RenderThermoviscous build/thermoviscous/sphere.json --output build/thermoviscous-pulse
script/RenderThermoviscous build/thermoviscous/sphere.json --input excitation.wav --fft-size 1024
```

The tool solves each positive-frequency FFT bin up to the band limit, caches solves by scene and executable hash, and constructs a band-limited receiver response. Mono input samples multiply the configured spatial velocity pattern; without an input WAV it generates a two-second pulse excitation. It conjugates the solver's frequency convention for NumPy, regularizes the cavity velocity-to-pressure pole through displacement, and uses linear convolution. Static/DC pressure offsets are excluded. This is Fourier synthesis of the same linear equations, not a time-stepping solver.

Outputs are physical `pressure.npy` in Pa, normalized `render.wav`, `transfer.npz`, and a report with playback gain, frequency spacing and impulse guard energy. Defaults are 8 kHz sampling, 512 FFT samples and a 3.6 kHz band limit with a cosine taper over its top 10%. The output includes the convolution tail. Increase `--fft-size` until the impulse guard check passes and compare successive sizes; passing the guard alone does not prove that narrow resonances are resolved. Mesh and frequency refinement remain necessary. Long-lived cavities can require many frequency solves.

## Paper benchmark

```sh
script/BenchmarkThermoviscous
script/BenchmarkThermoviscous --elements 246 --methods iterative direct1 direct4 direct10 --repetitions 20
```

The harness covers the paper's 100/1000 Hz sphere study, records analytic pressure error and separates boundary assembly, pressure solution and field recovery. All seven original meshes are covered, through 13,062 nodes. The 2,794-element mesh was deleted upstream to save space; the harness recovers it from commit `dcbc25949764493ca0dc78705782769bf74ac623` in the submodule history and records its hash. A shallow checkout must include that commit, or supply the original with `--mesh-2794`. The default is one repetition; the paper uses 20. Direct cases exceeding `--direct-memory-gib` are reported as skipped. These are reproducible local checks, not claims of matching published timings on different hardware.
