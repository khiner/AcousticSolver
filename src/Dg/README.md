# Curved tetrahedral acoustics

`DgGpu` advances homogeneous acoustics with rigid walls using strong–weak DG,
conservation-preserving weight-adjusted mass inversion, and five-stage LSERK4.
Geometry/operators are prepared in FP64; Metal propagation uses FP32 with fast math
disabled. Native mesh loading, source injection, scene integration and
frequency-dependent walls are outside its scope. Apple GPU family 7 or newer and
support for 1,024-thread groups are required.

After the [repository build setup](../../README.md#build), run from its root:

```sh
cmake --build build --target DgTest DgReference -j 6
build/DgReference prepare build/dg-inputs
build/DgTest build/dg-inputs/reduced/degree6/q12 build/dg-metal-check
```

Prepared tables can be reused. Preparation and Metal runs overwrite their generated
outputs. Optional positional arguments after the output directory are step count,
timestep and repeat count; defaults are `1024`, `2^-18 s` and `2`.

## Inputs and outputs

[Committed fixture documentation](../../tests/fixtures/dg/README.md) describes mesh
and reference provenance. Preparation generates cylinder initial states from an
analytic mode and reads receiver positions/element IDs from `mesh.json`. Upstream
imports preserve captured initial states and verify face maps and boundaries.

Mesh coordinates use FP64 `xyz.bin` in `[element][node][x,y,z]` order. Local face
maps come from reference nodes; self-mapped neighboring faces are rigid boundaries.
`dg::Tables` arrays are row-major FP32. Internal state layout is
`[element][node][pressure fluctuation, rho*c*vx, rho*c*vy, rho*c*vz]`.
The fixture's stationary `2 Pa` pressure offset is restored in output.

Each `DgTest` run writes:

- `final_q.bin`: terminal pressure/velocity, ordered `[element][field][node]`
- `receivers.bin`: receiver-major pressure samples
- `result.json`: settings, operator checks, energy history, timings and repeatability

Arrays serialize the FP32 solution as FP64 for scoring. Receiver sample `i` precedes
the step at `sample_start + i*dt`; there are `steps` samples per receiver. Metadata
records the FP32-rounded timestep. Repeated trajectories are compared byte for
byte in memory, with one copy saved. The report is written only after checks pass.

## Validation and independent references

```sh
build/DgReference check build/dg-reference-checks
build/DgReference validate build/dg-metal build/dg-inputs
```

The validator uses `DgTest` from the same build directory. It covers both physical
fixtures, repeatability, timestep refinement, long/larger trajectories, odd element
counts, alternate mass quadrature and invalid inputs. See [numerical gates and
limits](../../VALIDATION.md#curved-tetrahedral-dg).

Regenerate independent dense FP64 trajectories in new output directories:

```sh
build/DgReference reference degree6 build/dg-reference-degree6
build/DgReference reference radius_half build/dg-reference-full full_mass
```

The default is WADG; `full_mass` selects physical mass inversion. Each command
checks energy, conservation, analytic error and repeatability. WADG regeneration
also checks agreement with frozen FP64 records below `1e-10`, leaving them intact.
`DgReference score MESH RECORD NODES` reports analytic cylinder error;
`DgReference compare REFERENCE CANDIDATE TABLES [STRIDE]` compares state/receiver records.

## Upstream CUDA comparison

The comparison runner pins DTU/libParanumal at
`f08c22f83fd64605a634b94326f07cb88f00b0ef`. It records source, input, compiler and GPU
provenance. [The patch](../../script/dg_reference/upstream.patch) addresses Linux
portability, FP32 parsing/output, common clocks and capture/timing. Upstream
acoustics kernels, RK stages and fast-math flags remain unchanged. FP64 captures
geometry; timed upstream propagation uses FP32.

Generate the curved mesh with Gmsh 4.15.2:

```sh
python3 script/dg_reference/run_upstream_dg.py mesh build/cylinder-p6.msh
```

Copy the mesh, `run_upstream_dg.py` and `upstream.patch` together to a CUDA host.
The runner requires CUDA 12.8, a C++ compiler, Git, pkg-config, OpenMPI, HDF5,
Armadillo, BLAS/LAPACK development packages and Python NumPy. Use a new work directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_upstream_dg.py cuda \
  --work /workspace/dg-upstream --mesh /workspace/cylinder-p6.msh
```

Retrieve `runs/` and the pinned `upstream/nodes/` directory. On the Mac:

```sh
build/DgReference import-upstream build/dg-upstream/runs \
  build/dg-upstream/upstream build/dg-upstream-metal
build/DgTest build/dg-upstream-metal/cylinder/operator build/dg-cylinder-check \
  65536 0.00000762939453125 1
build/DgTest build/dg-upstream-metal/cylinder/operator build/dg-cylinder-timing \
  65536 0.00000762939453125 1 --benchmark
```

Import preserves initial fields and receiver interpolation, verifies the operator
checks, and writes `cube_rigid/operator` and `cylinder/operator`. The matched cube
uses 656 steps at `0.000030517578125 s`; the cylinder uses the values above.

`--benchmark` requires all positional arguments. It times synchronized propagation
in 256-step batches without intermediate CPU diagnostics, then checks terminal
energy/mean. A normal run checks these every 64 steps. Compare terminal/receiver
bytes between modes. Warm up, repeat in new output directories under nominal
thermal pressure, and report median `propagation_wall_seconds`. Final readback and
file output are excluded; upstream periodic receiver transfers remain timed.
The two methods have different spatial formulations, so assess numerical error
separately from speed.
