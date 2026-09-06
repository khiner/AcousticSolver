# Curved tetrahedral acoustics

`DgGpu` advances homogeneous acoustics with rigid walls using strong–weak DG,
conservation-preserving weight-adjusted mass inversion, and five-stage LSERK4.
Geometry and operators are prepared in FP64; native Metal propagation uses FP32.
The implementation accepts prepared tables. Native mesh loading, source injection,
scene integration, and frequency-dependent walls remain outside its scope.

From the repository root, with CMake and the macOS Metal toolchain available:

```sh
cmake -S . -B build
cmake --build build --target DgTest DgReference -j 6
build/DgReference prepare build/dg-inputs
build/DgTest build/dg-inputs/reduced/degree6/q12 build/dg-metal-check
```

Preparation generates the cylinder fixtures' initial states from the analytic
mode and reads receiver positions and element IDs from `mesh.json`. Upstream
imports preserve their captured initial states. Mesh coordinates use a single
FP64 `xyz.bin` in `[element][node][x,y,z]` order. Local face maps are generated
from reference nodes; self-mapped neighboring faces define rigid boundaries.
Upstream import verifies these maps and boundaries against the captured arrays.

Preparation and Metal runs overwrite their generated files when rerun. Prepared
tables in `build/dg-inputs/` can be reused across runs. Optional positional arguments after the output directory are step
count, timestep, and repeat count; defaults are `1024`, `2^-18 s`, and `2`.

## Method

With reference mass `M`, nodal-to-quadrature interpolation `V`, positive weights
`w`, and the reference-L2 projection `Jp` of the geometric Jacobian, mass inversion is

```text
T = V M^-1
A load = T^T [(w/Jp) .* (T load)]
```

Setup forms each fixed matrix `A` in FP64 from the prepared factors and uploads
it in FP32, avoiding quadrature products during propagation.
Only mass inversion uses `Jp`. Spatial geometry retains its original Jacobian
and normals. Projection preserves physical constant moments under consistent
quadrature; positivity is checked explicitly. The strong–weak spatial form is
needed for energy stability independently of this mass approximation. See
[Chan, Hewett and Warburton](https://arxiv.org/abs/1608.03836) and the
[validation contract](../../VALIDATION.md#curved-tetrahedral-dg).

The native kernels use 32-lane SIMD reductions, share volume matrices with their
transposes, and store symmetric face trace blocks. Planar faces use one scalar
matrix and a constant normal when all ten coefficient matrices reconstruct within
`1e-7` relative maximum error. Upload rejects inconsistent transpose or symmetry
identities. Receiver sampling shares the first RK stage's load dispatch.
Apple GPU family 7 or newer and support for 1,024-thread groups are required;
runtime validation covers Apple M5 Max. Fast math is disabled.

`dg::Tables` stores row-major FP32 arrays. State layout is
`[element][node][pressure fluctuation, rho*c*vx, rho*c*vy, rho*c*vz]`. Pressure
anchor subtraction preserves constant pressure exactly. The fixture's stationary
`2 Pa` offset is restored in output. Receiver sampling precedes each full step.

## Outputs and checks

`DgTest` checks mass and RHS applications against FP64, exact constant pressure,
modified energy, physical mean conservation, and repeated state/receiver bytes.
Each run writes three files:

- `final_q.bin`: terminal pressure and velocity, ordered `[element][field][node]`
- `receivers.bin`: receiver-major pressure samples
- `result.json`: run settings, operator checks, energy history, per-repeat metrics and timings, and repeatability status

The arrays serialize the FP32 solution as FP64 for independent scoring; this does
not increase propagation precision. Sample `i` is taken at `sample_start + i*dt`,
with `steps` samples per receiver. The report records the FP32-rounded timestep.
Repeated trajectories are compared byte for byte in memory; one copy is saved.
The report is written only after all checks pass.

## Regression suite

After preparing inputs, run the complete validation suite:

```sh
build/DgReference validate build/dg-metal build/dg-inputs
```

The validator uses `DgTest` from the same build directory and checks both physical
fixtures, repeated runs, half timesteps, long trajectories, larger working sets,
odd element counts,
original quadrature, and invalid inputs. It writes records and provenance under
`build/` and requires nominal macOS thermal pressure. Numerical gates are defined
in [validation](../../VALIDATION.md#curved-tetrahedral-dg).

## Upstream CUDA comparison

The README table measures native DTU/libParanumal CUDA at commit
`f08c22f83fd64605a634b94326f07cb88f00b0ef`. `run_upstream_dg.py` builds disposable
FP64 and FP32 copies and records source, kernel, input, compiler, and GPU provenance.
FP64 is used only to capture geometry; the timed upstream runs use FP32.
[The patch](../../script/dg_reference/upstream.patch) fixes Linux portability,
FP32 receiver parsing and output, selects a common timestep/step count,
and adds synchronized timing and raw capture. It leaves the acoustics kernels,
RK stage routine, and upstream CUDA fast-math flags unchanged.

Generate the curved mesh once with Gmsh 4.15.2 and transfer that exact file,
`run_upstream_dg.py`, and `upstream.patch` together to the CUDA host:

```sh
python3 script/dg_reference/run_upstream_dg.py mesh build/cylinder-p6.msh
```

On Ubuntu with CUDA 12.8, a C++ compiler, Git, pkg-config, OpenMPI, HDF5,
Armadillo, BLAS/LAPACK development packages, and Python NumPy:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_upstream_dg.py cuda \
  --work /workspace/dg-upstream --mesh /workspace/cylinder-p6.msh
```

Use a new work directory. The runner clones upstream, builds both precisions,
and writes mesh probes, a 64-step warmup, four timed records per case, and
`runs/timings.json`. Retrieve `runs/` and the pinned `upstream/nodes/` directory.
On the Mac, prepare matching Metal operators using those captures:

```sh
build/DgReference import-upstream build/dg-upstream/runs \
  build/dg-upstream/upstream build/dg-upstream-metal
```

Preparation preserves upstream FP32 initial fields and receiver interpolation,
uses FP64 geometry from the same mesh, and verifies the existing mass/RHS and
reduced-quadrature tolerances. It creates `cube_rigid/operator` and
`cylinder/operator`. Run the cube with `656 0.000030517578125` and cylinder with
`65536 0.00000762939453125` as the step count and timestep:

```sh
build/DgTest build/dg-upstream-metal/cylinder/operator build/dg-cylinder-check \
  65536 0.00000762939453125 1
build/DgTest build/dg-upstream-metal/cylinder/operator build/dg-cylinder-timing \
  65536 0.00000762939453125 1 --benchmark
```

`--benchmark` requires all positional arguments. It times synchronized propagation
without intermediate state readback or CPU diagnostics, then checks the terminal
energy and mean. A separate default run checks these every 64 steps. Compare
its terminal/receiver bytes with every timed run. Warm up first, collect four
runs in new output directories, and take the median `propagation_wall_seconds` from `result.json`’s `runs` array.
Require nominal OS thermal pressure before and after each run, allowing cooling
between runs; the batched `repeats` argument can otherwise heat the machine.
The timing excludes the final receiver/state readback and disk output on both
backends; upstream's periodic receiver transfers remain inside its timed loop.

The benchmark workloads and numerical differences are documented in
[validation](../../VALIDATION.md#upstream-native-cuda-benchmark); measured times
are in the [root README](../../README.md#curved-tetrahedral-dg).

## Independent reference checks

`DgReference` uses vendored Eigen and Accelerate to assemble full-mass and
conservation-preserving WADG operators in FP64. Preparation checks the dense and
factored applications in FP64 and FP32, plus the selected mass quadrature's
spectrum and constant moments. Its analytic cylinder solution and physical error
integral check spatial and temporal accuracy independently of the DG operator.
Run the analytic and precision-comparison checks with:

```sh
build/DgReference check build/dg-reference-checks
```

Regenerate independent dense FP64 trajectories with:

```sh
build/DgReference reference degree6 build/dg-reference-degree6
build/DgReference reference radius_half build/dg-reference-full full_mass
```

The default form is `wadg`; the final `full_mass` argument selects physical mass
inversion. Each command checks energy, conservation, analytic error, and two
identical trajectories. WADG regeneration also checks agreement with the frozen
FP64 records below `1e-10`. Output directories must be new; these commands leave
the frozen fixtures intact. Dense FP64 stepping remains independent of Metal's
factored mass application and pressure-anchor subtraction.

`DgReference score MESH RECORD NODES` reports analytic cylinder error;
`DgReference compare REFERENCE CANDIDATE TABLES [STRIDE]` compares complete state
and receiver records. Only upstream CUDA build/run orchestration and Gmsh mesh
generation require Python.
