# Validation

The Metal port is validated against golden outputs from the reference CUDA implementation
(RTX 4090, CUDA 12.8, gcc, Linux — `script/generate_golden_outputs.sh`, stored in
`golden/`). `script/ValidateGolden` runs every scene and compares the listener pressure
output sample-for-sample. `wav/` holds paired renderings for listening.

## Key numerics decisions

- All Metal kernels compile with **fast-math disabled**, matching nvcc's default IEEE
  behavior — this is what makes bit-exact GPU comparison possible.
- Every floating-point expression matches the CUDA reference operation-for-operation, and
  every optimization round was gated on byte-identical listener outputs vs the previous
  build plus the unchanged bit-exact prefix vs CUDA below.
- Three deliberate output-observable changes, all validated:
  1. **Fresh-cell least squares per connected component.** The reference's global
     minimum-norm system is block-diagonal across connected components, so solving per
     component is identical in exact arithmetic and differs only in float rounding, while
     being asymptotically cheaper and parallel. `ACOUSTIC_GLOBAL_LSTSQ=1` restores the
     reference-bit-identical global solve.
  2. **apply_shader write-race fix.** In the reference, a velocity face claimed by both a
     velocity-blend shader and a point-source force (5,265 collisions in FillerUp) is
     raced by two threads, making output **nondeterministic run-to-run** (reproduced on
     both the CUDA structure and the original port). Blend and force points now dispatch
     as two ordered ranges; non-colliding faces are bit-unaffected. The fix moved
     FillerUp from envelope-level to full waveform agreement with the golden (rel L2
     3.4e-2 → 2.0e-4) and tightened HandShake (6.5e-2 → 2.5e-2).
  3. **Explicit bubble mass-matrix inverses.** The coupled-bubble solve applies
     precomputed explicit inverses (`dpotrf`/`dpotri`, chained by SPD Schur updates — see
     Performance) instead of factor-and-substitute. Same linear systems, different
     operation order: bubble-scene outputs differ at rel L2 ~1e-6, five orders below the
     CUDA-golden comparison level (byte-identical for GlassPour, below float32 output
     quantization). A derived inverse is always exactly one update from a fresh
     factorization, so rounding error never compounds. The inline fallback path factors
     and substitutes exactly as the reference does.

## What "matching" means across platforms

Three regimes, in decreasing strictness:

1. **Bit-exact.** CupPhone matches CUDA bit-for-bit for 4.66 of 8 simulated seconds —
   until the first moving-geometry batch — including through the fused-kernel pipeline.
   TalkFan and Trumpet match to rel L2 ~1e-4..1e-5 (−73 to −95 dB) over 10+ seconds with
   animated geometry.
2. **Ulp-accumulation drift.** CPU-side per-batch math (quaternion slerp, fresh-cell
   least squares, modal ODE stepping) picks up last-ulp ARM-vs-x86 differences from libm
   and Eigen reduction orders, accumulating as phase drift: LegoDrop −59 dB, SpollingBowl
   −44 dB, correlation ≥ 0.99998, matching spectra.
3. **Phase decorrelation (chaotic CPU paths).** The coupled-bubble ODE amplifies ulp
   differences to macroscopic divergence. **Control experiment**: the CPU-only FluidSound
   solver alone, ARM/clang vs x86/gcc on identical GlassPour input, diverges at rel L2 =
   0.40, corr = 0.92 — the same magnitude as the full Metal-vs-CUDA comparison (0.37,
   0.93). Envelope (RMS within ±6%) and magnitude spectrograms remain matched, so those
   are the meaningful invariants, graded `OK(env)`.

A structural port bug would fail all three gates at once (wrong energy, wrong spectrum,
zero bit-exact prefix); nothing does.

## Results (full suite)

Established 2026-08-15, reproduced to all displayed digits after every optimization round
since:

| Scene | Sources | Verdict | rel L2 | corr | RMS ratio | spec err |
|---|---|---|---|---|---|---|
| CupPhone | Speaker + Occluder | OK | 1.3e-5 | 1.000000 | 1.0000 | 7.5e-6 |
| TalkFan | Speaker ×2 + Occluders | OK | 1.9e-5 | 1.000000 | 1.0000 | 1.1e-5 |
| FillerUp | Point + Density | OK | 2.0e-4 | 1.000000 | 1.0000 | 2.4e-4 |
| Trumpet | Speaker + Occluder | OK | 2.3e-4 | 1.000000 | 1.0000 | 1.1e-4 |
| LegoDrop | Modal | OK | 1.2e-3 | 0.999999 | 0.9999 | 6.1e-4 |
| SpollingBowl | Modal | OK | 6.3e-3 | 0.999980 | 1.0000 | 3.0e-3 |
| HandShake | Point + Occluders | OK(env) | 2.5e-2 | 0.999683 | 1.0004 | 1.8e-2 |
| 2016Pour | Bubbles | OK(env) | 3.3e-1 | 0.946 | 0.9937 | 2.2e-1 |
| GlassPour | Bubbles | OK(env) | 3.7e-1 | 0.931 | 0.9909 | 2.5e-1 |
| PaddleSplash | Bubbles | OK(env) | 4.7e-1 | 0.894 | 1.0149 | 3.2e-1 |

The bubble scenes sit at exactly the divergence level the CPU-only control experiment
predicts, with energy within ±1.5%.

## Bugs found during validation

- **apply_shader write race** (reference + original port): see above. Detected because
  the port is otherwise bit-reproducible run-to-run and FillerUp wasn't.
- **Upstream FluidSound UB** (`Integrators.h`): `Solver::~Solver()` deletes derived
  integrators through a base pointer with no virtual destructor. gcc compiled it
  benignly; Homebrew clang 22 at -O2 compiles the delete to a trap, which crashed
  GlassPour at teardown and truncated the final batch. Patched in our vendored copy
  (`LOCAL PATCH`); upstream is archived.

## Performance

Seven optimization rounds took the full suite from 1170s (initial straight translation)
to 334s on an M5 Max, every scene beating the RTX 4090 reference wall-clock (per-scene
table in the README). Below is the durable outcome: how the solver is structured, what
this hardware was measured to want, and how to benchmark it — all under the
bit-exactness regime above.

### GPU solver

- **One fused full-grid kernel per step** (`StepVelocityPressure`): the velocity update
  at step s and the pressure update at s+1 in one pass, with P/velocity ping-pong
  buffers (neighbor reads would race in-place writes). Each thread recomputes its
  lower-neighbor face velocities bit-identically instead of communicating. Listener
  sampling is folded in.
- **Byte cell states**: beta is always 0, 1, or the cubic smoothstep of the step's blend
  time, so a state byte plus the step time reproduces it in-register — one uint8 stream
  replaces a float beta field, a cell array, and a transition list.
- **Shell-packed PML split fields** (`PsiIndex`): px/py/pz are touched only in the PML
  shell (45–58% of cells at these grid sizes), and full-grid storage wasted most of each
  cache line in the 8-wide x-PML strips. They live in six dense slabs, each contiguous
  along x. `Px/Py/Pz` (full-grid) serve only as fresh-cell acceleration scratch — the
  two roles use disjoint cell regions.
- **Boundary application, two paths, runtime-tuned**: standalone `ApplyShader`
  dispatches (blend range then force range, giving dual-claimed faces a defined order)
  vs folding into the velocity kernels via a face-mask lookup with per-tile flags.
  Folding removes dispatches but costs fixed whole-grid register pressure, so probe
  batches lock the faster path per scene (`[apply-path]`). Probing re-opens when the
  shader-point count leaves [½×, 2×] of its value at lock, and near-ties with force
  points prefer folded (its cost is fixed, plain's scattered applies degrade with load).
- **Per-scene threadgroup shape**: 128×4×1 for Nx ≥ 88, 32×4×4 below (wide tiles lose on
  small grids to partial rows). The tile-flag lookup reads its extents from
  `FdtdBatchParams`, so the shape is a host choice.
- Small tables (PML LUTs, listener cell ids) bind in the `constant` address space.

### CPU/GPU pipeline

- The GPU executes batch N while the CPU prepares batch N+1, one sync per batch at the
  fresh-cell velocity solve. CPU-rewritten GPU inputs are double-buffered.
- The batch's FDTD command buffer (shader re-init + all steps) is encoded and committed
  *before* the sync, headed by an `MTLSharedEvent` wait the host signals right after its
  fresh-cell velocity writes. The prologue buffer commits as soon as it is encoded.
- The fresh-cell solve is split: `AnalyzeFreshCells` (components, system assembly,
  factorization — geometry-only) runs pre-sync overlapped with the GPU;
  `SolveFreshCells` (right-hand side, triangular solves, scatter) is all that remains
  inside the window.
- Listener samples are double-buffered so file IO happens after the next batch is
  released. The sync blocks on the second-to-last command buffer and spins out the last,
  hiding `waitUntilCompleted`'s ~100 µs wake.

### Bubble solver (FluidSound path)

- Worker threads replay the solver's event bookkeeping ahead of consumption, building
  and inverting each event interval's mass matrix (`dpotrf`/`dpotri`); the main thread's
  RK4 stages apply two `cblas_dgemv` calls plus a blend. On AMX, GEMV is ~10× faster
  than a triangular-solve pair at these sizes — the whole case for explicit inverses.
- **Chained endpoint inverses**: interval k+1's start matrix is interval k's end matrix
  minus dead oscillators plus born ones (mean churn ~1.5 rows), so its inverse derives
  by SPD Schur delete-downdate and bordered extension in O(N²r), halving the O(N³)
  factorization load. Non-derivable successors fall back to a fresh inversion.
- **Byte-budgeted lookahead**: workers run ahead until they hold 1.5 GB of unconsumed
  inverses (events cluster — fixed lookaheads stall in dense stretches). Worker count
  defaults to 4 (`ACOUSTIC_BUBBLE_WORKERS` overrides).

### Modal solver

- Per-mode IIR coefficients compute once (they depend only on eigenvalues, material,
  timestep), impulse queries scan a binary-searched timestamp window, history rotates by
  pointer swap.
- Mode-to-boundary rows are keyed by grid-face identity and reused across batches when
  the rest-frame query point and normal are bit-identical to the snapshot row. Copies
  run in column sweeps, fresh rows land via a blocked transpose, and when every row hits
  at its own position the previous end-matrix GPU buffer is rebound with no copy.

### CPU batch prep

- Every per-batch sweep is bounded by rasterization/cavity cell boxes (`CellBox`) and
  iterates in ascending cell order — several sweeps' float rounding depends on that
  order, so it is load-bearing.
- Flat lookup tables and epoch-stamped face sets instead of sets/maps, parallel
  rasterization and closest-point queries, a data-oriented AABB tree, time-windowed
  impulse queries.
- Parsers for large inputs (.obj meshes, Density fields, bubble files) use
  `strtof`/`from_chars`-class conversions — correctly rounded like the stream extractors
  they replace, so parsed values are bit-identical.

### Measured limits (Apple M5 Max) — do not retry without new hardware

- The fused FDTD kernel runs within ~2% of a pure stream-copy floor at scene grid sizes.
  Redundant halo arithmetic is free; threadgroup-memory staging does not help.
- **2-step temporal blocking is slower** (bit-exact variant measured): extra cached
  loads and barriers outweigh the halved traffic.
- **Multi-cell-per-thread vectorization is much slower** (~50% at 88³): this GPU wants
  many light threads.
- **Concurrent tile-split boundary application gains nothing**: serial-encoder
  inter-dispatch cost is under ~1 µs, and a tile-list variant loses at high point
  counts.
- **AMX saturates at ~660 `dpotrf`+`dpotri`/s aggregate regardless of thread count**;
  extra workers (any QoS) stretch every call including the main thread's GEMVs — 4
  workers is the optimum. `cblas_dsymv` is ~3× slower than `dgemv`, and
  `dtrsm`/`dtrtri`-based alternatives lose to `dpotri`. One fresh inversion per distinct
  event time is the algorithmic floor.
- **Warm-start seeding of closest-point queries flips ulp-level ties** (a computed
  candidate can round outside its box on the query-facing side) — output-observable, and
  not faster anyway.
- GPU execution is **memory-bandwidth-coupled to concurrent CPU work**: a heavy CPU job
  inflates `gpu/exec` itself (measured +16 µs/step under a parallel build vs +4 µs
  rested), not just wall clock.

### Instruments and toggles

- `ACOUSTIC_PROFILE=1`: per-phase wall time at exit — `gpu/exec` (total GPU busy) with
  its `gpu/exec_fdtd` / `gpu/exec_prologue` split, `gpu/idle` (gaps between command
  buffers on the GPU timeline — the structural pipeline-bubble measure, robust to
  machine load), `gpu/wait`, `startup/scene_load`, and the `[apply-path]` and
  `[bubble-pipeline]` lines.
- `ACOUSTIC_GLOBAL_LSTSQ=1`: reference-bit-identical global fresh-cell solve.
- `ACOUSTIC_LAZY_COMMIT=1`: commit prologue/FDTD buffers at the sync instead of eagerly.
- `ACOUSTIC_BUBBLE_WORKERS=<n>`: bubble inverse-pipeline worker count.

### Benchmarking method

- Runs are bit-reproducible, so `cmp` of listener `.bin` outputs is the exactness gate.
- Time only on a quiet machine (see bandwidth coupling above), A/B only interleaved with
  a baseline binary built from a pinned worktree — kernels compile at runtime from
  `src/FDTD/`, so a saved binary is not a snapshot without its matching MSL sources.
- Thermal drift (GPU and AMX) biases the second run of a pair. Re-pair suspected
  regressions in reversed order before believing them.
- Back-to-back suite totals inflate GPU-heavy scenes ~10% vs rested standalone runs; use
  them for verdicts and coarse tracking, not attribution.

## Rerunning

```
script/Build                       # build (Homebrew LLVM, C++23)
script/ValidateGolden              # all scenes with golden data
script/ValidateGolden GlassPour    # one scene
ACOUSTIC_PROFILE=1 build/AcousticSolver Scenes/<scene>/config.json   # phase timing
ACOUSTIC_GLOBAL_LSTSQ=1 ...        # reference-bit-identical fresh-cell solve
script/generate_golden_outputs.sh root@<pod-ip> <port>   # regenerate goldens (CUDA host)
```
