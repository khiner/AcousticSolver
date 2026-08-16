# Metal Porting Resources (CUDA → metal-cpp + MSL)

Researched 2026-08-15. Items that could not be fully verified are flagged inline.

## Official Apple documentation

- **Metal Shading Language Specification (PDF)** — https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
  Canonical MSL reference (stable URL, currently v4.x). MSL is C++14/17-based, so CUDA `__device__` code translates with modest syntax changes. Key chapters: address spaces (`device`, `threadgroup`, `constant`), thread attributes (`[[thread_position_in_grid]]` etc.), SIMD-group functions, atomics, floating-point conformance tables (`fast::` vs `precise::` math).
- **metal-cpp** — https://developer.apple.com/metal/cpp/ and https://github.com/apple/metal-cpp
  Official header-only C++ interface (C++17). Main gotcha: manual `NS::Object` retain/release (no ARC).
- **WWDC22 "Program Metal in C++ with metal-cpp"** — https://developer.apple.com/videos/play/wwdc2022/10160/ — project setup, object lifetime, interop for CMake-style (non-Xcode) hosts.
- **WWDC22 "Scale compute workloads across Apple GPUs"** — https://developer.apple.com/videos/play/wwdc2022/10159 — the most relevant session for this port: minimizing GPU-timeline gaps, pipelining, `MTLDispatchTypeConcurrent`, compute counters. The "gaps between dispatches" material is exactly the failure mode a naive one-dispatch-per-timestep FDTD loop hits.
- **Tech Talk "Metal Compute on MacBook Pro"** — https://developer.apple.com/videos/play/tech-talks/10580/ — Apple's closest thing to an HPC-porting guide: dispatch sizing, unified-memory bandwidth, occupancy.
- **Tech Talk "Learn performance best practices for Metal shaders"** — https://developer.apple.com/videos/play/tech-talks/111373/ — register pressure, occupancy, memory access patterns.
- **"Porting your Metal code to Apple silicon"** — https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon — authoritative on unified memory, storage modes, TBDR differences. Pair with WWDC20 https://developer.apple.com/videos/play/wwdc2020/10631/.
- **Metal Feature Set Tables (PDF)** — https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf — per-GPU-family limits: 1024 max threads/threadgroup, **32 KB threadgroup memory** (vs CUDA's 48–228 KB — moot for WaveBlender, which uses no shared memory).
- **"Creating threads and threadgroups"** — https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups — `dispatchThreads` (non-uniform threadgroups, removes `if (i < N)` edge guards) vs `dispatchThreadgroups` (exact `<<<grid, block>>>` analog).
- **Tooling**: GPU frame capture for compute via `MTLCaptureManager` — https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-in-xcode; WWDC21 tools session — https://developer.apple.com/videos/play/wwdc2021/10157/; WWDC20 GPU counters — https://developer.apple.com/videos/play/wwdc2020/10603/. The shader debugger steps through compute kernels per-thread (the Nsight/cuda-gdb equivalent).
- Archived-but-clear: Metal Programming Guide and Best Practices Guide (documentation archive) — pre-Apple-silicon, cross-check hardware claims. https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Introduction/Introduction.html

### metal-cpp + CMake

No official CMake support. Community pattern: interface target for headers, link `Metal`/`Foundation`/`QuartzCore`, custom command `.metal` → `.air` → `.metallib` via `xcrun -sdk macosx metal`. Verified exemplars:
- https://github.com/pr0g/metal-cpp-cmake
- https://github.com/arauzca/cmake-metal-cpp
- https://github.com/LeeTeng2001/metal-cpp-cmake
- https://github.com/Valken/metal-cmake
- https://github.com/MattGuerrette/Metal (idiomatic metal-cpp example code)

## CUDA → Metal concept mapping

grid→grid, block→threadgroup, thread→thread, warp(32)→SIMD-group(32), `__shared__`→`threadgroup`, `__syncthreads()`→`threadgroup_barrier(mem_flags::mem_threadgroup)`, shuffles→`simd_shuffle*`, streams→command queues/buffers, events→`MTLEvent`/`MTLSharedEvent`, `cudaMemcpy`→unnecessary with `MTLStorageModeShared` on Apple silicon. Indexing: `threadIdx`→`[[thread_position_in_threadgroup]]`, `blockIdx`→`[[threadgroup_position_in_grid]]`, `blockDim`→`[[threads_per_threadgroup]]`, and `[[thread_position_in_grid]]` (uint3, natively 3D) replaces the `blockIdx*blockDim+threadIdx` idiom — convenient for WaveBlender's 8×8×8-block 3D grid kernels.

- **Gebraad & Fichtner, "Seamless GPU acceleration for C++-based physics with the Metal Shading Language on Apple's M-series chips"** — https://arxiv.org/abs/2206.01791 (Seismological Research Letters 94(3), 2023). **The closest published precedent to this exact port**: a C++ elastic wave-equation FDTD solver accelerated with MSL on M1, benchmarking finite-difference stencils, ~10× over CPU, unified memory eliminating explicit memory management. Read before writing code.
- **Metal.jl kernel docs** — https://metal.juliagpu.org/stable/usage/kernel/ — well-maintained CUDA↔Metal terminology mapping (mirrors CUDA.jl's API).
- https://github.com/philipturner/metal-usm — pointer/address-space differences from CUDA documented via a USM-emulation project.
- https://github.com/xmartlabs/gpgpu-comparison — side-by-side CUDA/Metal implementations (flag: older repo).
- Flag: no single official CUDA→Metal guide exists; the above assembles the mapping from primary sources.

## BLAS (replacing cuBLAS Sgemv/handles)

- **MPSMatrixMultiplication / MPSMatrixVectorMultiplication** — https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication — near drop-in for the cuBLAS calls; encodes onto a normal `MTLCommandBuffer` so it interleaves with FDTD kernels. **Not deprecated** (verified 2026-08-15; only MPS ray-tracing classes are). Metal 4's `MTLTensor` path (WWDC25 https://developer.apple.com/videos/play/wwdc2025/205/) is the ML-oriented future, but plain MPS remains appropriate for a single GEMV.
  - Tutorial on `MPSMatrix` row-bytes layout: https://machinethink.net/blog/mps-matrix-multiplication/
- **Custom-kernel alternative**: MLX "steel" GEMM kernels (https://github.com/ml-explore/mlx, `mlx/backend/metal/kernels/steel/`) — best open-source reference for `simdgroup_matrix` 8×8 tile GEMM. For WaveBlender's small matrices (N_boundary_points × N_modes/bubbles), a trivial custom kernel may beat MPS encoding overhead — and folding the projection into the shader-apply gather kernel is worth considering before reaching for GEMM at all.

## CPU-side dense linear algebra (Accelerate)

Added 2026-08-16 (perf round 3): the bubble-solver rework precomputes per-event-interval mass-matrix inverses and applies them with GEMV, leaning on Accelerate's AMX/SME-backed BLAS/LAPACK.

- **Accelerate BLAS/LAPACK** — https://developer.apple.com/documentation/accelerate/blas-library and https://developer.apple.com/documentation/accelerate — Apple's LAPACK/CBLAS, dispatching to the per-cluster matrix coprocessor (AMX/SME). Define `ACCELERATE_NEW_LAPACK` for the modern headers (https://developer.apple.com/documentation/accelerate/using-the-latest-lapack-interfaces).
- Measured locally on M5 Max (double precision, N≈958): `cblas_dgemv` ~10 µs (≈6–10× Eigen's NEON gemv), `dpotrf` 1.1 ms, `dpotri` 2.5 ms, `cblas_dtrsv` ~50 µs (triangular solves gain little from the matrix units — the reason explicit inverses + GEMV beat substitution here). Aggregate potrf+potri throughput saturates at ~660 inversions/s regardless of thread count (the matrix units are shared per cluster; more threads only stretch each call), and low-QoS threads do not add throughput — they contend for the same units.
- **corsix's AMX notes** — https://github.com/corsix/amx — unofficial but standard reference on the AMX coprocessor's per-cluster sharing, consistent with the saturation behavior measured here.

## Precision and determinism

- **No float64 on Apple GPUs — confirmed.** MSL has no `double` for device code (corroborated: https://developer.apple.com/metal/jax/, https://developer.apple.com/forums/thread/709663, and the existence of the emulation library https://github.com/philipturner/metal-float64). Not a blocker here: WaveBlender's device code is all fp32; doubles live in CPU-side ODE solvers (FluidSound, ModalSound) that port unchanged.
- **Fast-math is ON by default in Metal** — the opposite of nvcc, and the single biggest validation trap. Control via `MTLCompileOptions.mathMode` (`safe`/`relaxed`/`fast` — https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode; replaces `fastMathEnabled`), or `-fno-fast-math` to `xcrun metal`. Use `safe` for validation builds, then measure whether `fast` shifts results beyond tolerance. Precedent: MLX added per-kernel `math_mode` for exactly this (https://github.com/ml-explore/mlx/pull/3728).
- **FMA/contraction**: nvcc contracts to FMA by default (`--fmad=true`); Metal's contraction choices differ even in precise mode → **bit-for-bit CUDA-vs-Metal equality is unattainable**. Plan tolerance-based validation (relative L2/L∞ over N timesteps vs a CUDA or CPU fp32 reference with controlled FMA). Background: https://simonbyrne.github.io/notes/fastmath/.
- `half` is fully supported — potentially interesting for auxiliary fields later, but not for pressure/velocity state at 100k steps without a dispersion-error study.

## Exemplar open-source projects (verified)

- **llama.cpp** (`ggml/src/ggml-metal/`) — https://github.com/ggml-org/llama.cpp — most battle-tested Metal compute backend: runtime `.metal` library compilation, batching hundreds of dispatches per command buffer, pure C/C++ host without Xcode.
- **MLX (Apple)** — https://github.com/ml-explore/mlx — highest-quality idiomatic MSL + C++ host code (GEMM, reductions, scheduling). Its custom-kernel API (https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) is also a fast Python harness for prototyping individual ported kernels before committing to the metal-cpp host.
- **metal-benchmarks** — https://github.com/philipturner/metal-benchmarks — de facto Apple-GPU microarchitecture reference (latencies, caches, occupancy); use to set stencil-kernel bandwidth expectations.
- **BabelViscoFDTD** — https://github.com/ProteusMRIgHIFU/BabelViscoFDTD — production viscoelastic **staggered-grid FDTD wavesolver with CUDA, OpenCL, and Metal backends** (transcranial ultrasound); published evidence (BabelBrain, https://pubmed.ncbi.nlm.nih.gov/37155375/) that M1 Max beats discrete GPUs on this workload class via unified memory. Flag: Metal path uses a Swift interface layer (predates metal-cpp maturity) — study its kernel organization and CUDA/Metal source-sharing macros, not its host bindings.
- Smaller: https://github.com/Pierre-Joly/SPH-Fluid-Metal-GPU.

## Dispatch overhead: batching ~100k timesteps

Metal's cost hierarchy: per-command-buffer commit ≫ per-encoder > per-dispatch. Strategy for WaveBlender's 3 dispatches × ~1000 steps per 10 ms batch:

1. Encode an entire batch (thousands of dispatches) into **one command buffer** with a single serial `MTLComputeCommandEncoder`.
2. Keep 2–3 command buffers in flight so CPU encoding overlaps GPU execution (WWDC22 10159 pipelining guidance).
3. Per-command-buffer round-trip latency is ~0.2 ms order-of-magnitude (flag: unverified secondhand figure) — commit-and-wait per timestep would add ~20 s overhead per simulated second, so batching is mandatory, matching CUDA's async-stream behavior.
4. **Indirect command buffers** — https://developer.apple.com/documentation/metal/indirect-command-encoding — encode the fixed per-timestep kernel sequence once, re-execute repeatedly. Evaluate only if plain batching leaves CPU encoding on the critical path.

**GPU timing**: `MTLCommandBuffer.gpuStartTime/gpuEndTime` for whole-batch throughput; `MTLCounterSampleBuffer` for per-kernel timings (units pitfalls: https://feresignum.com/resolving-metal-gpu-timers/; production example in Blender Cycles https://projects.blender.org/blender/blender/pulls/121208); Instruments Metal System Trace for timeline-gap analysis with no code changes.

## Cross-cutting notes

- Host↔device transfers per batch (shader data up, listener audio down) mostly disappear: `MTLStorageModeShared` buffers are the same physical memory; the remaining cost is synchronization, not copying. Eigen geometry code can write directly into shared `MTLBuffer` memory.
- WaveBlender's kernels use no shared memory, atomics, or streams, so the 32 KB threadgroup-memory limit and threadgroup-size limits don't bind; the 8×8×8 = 512-thread blocks fit the 1024 limit as-is.
