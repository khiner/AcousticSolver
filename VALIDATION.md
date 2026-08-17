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

Nine optimization rounds took the full suite from 1170s (initial straight translation) to
316s on an M5 Max, every scene beating the RTX 4090 reference wall-clock. The README has
the per-scene table and a summary of what changed; each mechanism is documented where it
lives in the code.

The solver now sits at measured hardware limits: the fused FDTD kernel sustains 752–766 GB/s
at 80³ and above, which is this machine's DRAM roof (a pure streaming copy of the same
element count reaches 725–741 GB/s there), and the coupled-bubble ODE is bound by the same
roof on its inverse matrix reads. Those two account for ~95% of the suite.

Every round was gated on byte-identical listener output against the previous build, so the
verdicts above hold unchanged across all of them.

### Instruments and toggles

- `ACOUSTIC_PROFILE=1`: per-phase wall time at exit — `gpu/exec` (total GPU busy) and its
  `gpu/exec_fdtd` / `gpu/exec_prologue` split, `gpu/idle` (gaps between command buffers on
  the GPU timeline, the pipeline-bubble measure) and its `gpu/idle_fdtd` /
  `gpu/idle_prologue` split, `cpu/sync_window`, `gpu/wait`, `startup/scene_load`, and the
  `[apply-path]`, `[shader-faces]` and `[bubble-pipeline]` lines.
- `ACOUSTIC_APPLY_PATH=plain|folded`: pin the boundary-application path.
- `ACOUSTIC_GLOBAL_LSTSQ=1`: reference-bit-identical global fresh-cell solve.
- `ACOUSTIC_LAZY_COMMIT=1`: commit prologue/FDTD buffers at the sync instead of eagerly.
- `ACOUSTIC_BUBBLE_WORKERS=<n>`: bubble inverse-pipeline worker count.

### Benchmarking method

- Runs are bit-reproducible, so `cmp` of listener `.bin` outputs is the exactness gate.
- Time only on a quiet machine, and mind that thermal drift dominates cross-run GPU timing
  here: a GPU-bound scene measures 43s cool and 53s heat-soaked. Start every run from an
  idle GPU and compare against recorded cool numbers, or measure inside one run.
- Kernels compile at runtime from `src/FDTD/`, so a saved binary is not a snapshot without
  its matching MSL sources.

## Rerunning

```
script/Build                       # build (Homebrew LLVM, C++23)
script/ValidateGolden --profile    # render every scene once: verdicts + wall + phase timing
script/ValidateGolden --score      # re-score build/*.bin — no rendering
script/ValidateGolden GlassPour    # one scene (either mode)
ACOUSTIC_GLOBAL_LSTSQ=1 ...        # reference-bit-identical fresh-cell solve
script/generate_golden_outputs.sh root@<pod-ip> <port>   # regenerate goldens (CUDA host)
```

**Render each scene at most once per session.** Scoring is a pure function of the listener
bytes and the golden bytes, so any run leaves what a later `--score` needs, persisted in
`build/validation.json`. And outputs byte-identical to a build whose metrics are recorded
here have identical metrics by construction — `cmp` settles that, re-rendering does not.
