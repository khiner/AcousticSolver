# Validation

The Metal port is validated against listener outputs from the reference CUDA implementation
(RTX 4090, CUDA 12.8, gcc, Linux — `script/generate_golden_outputs.sh`, committed under
`gen/cuda/`). `script/ValidateGolden` runs every scene and compares the listener pressure
output sample-for-sample. `gen/wav/{cuda,metal}/<scene>/<rate>.wav` holds paired renderings for listening, filed by rate
like the references, so a retimed scene's previous rendering stays listenable beside its new one.

The reference outputs are committed, so scoring works from a fresh clone without a CUDA host.
They are filed by the rate they were generated at, so a retimed scene compares against a
reference at the same rate. The scene data is not committed (~12 GB) — run
`script/FetchScenes` first. `Scenes/` is fetched and checksummed, so anything we change about
a scene lives in `config/<scene>.json` rather than being edited in place there.

## Key numerics decisions

- All Metal kernels compile with **fast-math disabled**, matching nvcc's default IEEE
  behavior — this is what makes bit-exact GPU comparison possible.
- Every floating-point expression matches the CUDA reference operation-for-operation. Changes
  are gated on byte-identical listener outputs against the previous build, plus the unchanged
  bit-exact prefix against CUDA below.
- Four deliberate output-observable changes, all validated:
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
     FillerUp from envelope-level to full waveform agreement with the reference (rel L2
     3.4e-2 → 2.0e-4) and tightened HandShake (6.5e-2 → 2.5e-2).
  3. **Explicit bubble mass-matrix inverses.** The coupled-bubble solve applies
     precomputed explicit inverses (`dpotrf`/`dpotri`, chained by SPD Schur updates — see
     Performance) instead of factor-and-substitute. Same linear systems, different
     operation order: bubble-scene outputs differ at rel L2 ~1e-6, five orders below the
     CUDA-reference comparison level (byte-identical for GlassPour, below float32 output
     quantization). A derived inverse is always exactly one update from a fresh
     factorization, so rounding error never compounds. The inline fallback path factors
     and substitutes exactly as the reference does.
  4. **FDTD timestep raised toward the CFL limit** on TalkFan, Trumpet and 2016Pour. The
     paper sets the step rate from the audio rate rather than from the grid — *"our FDTD
     timestep size is often higher (88.2 kHz and above) than our target grid resolutions—as
     determined by the CFL condition"* (§6.1) — leaving those three at Courant numbers of
     0.357–0.389 against the 7-point scheme's limit of 1/√3 = 0.577. They now run at
     0.519–0.536: 1.34× fewer steps for the speaker scenes, 1.5× for 2016Pour.

     **This lowers numerical dispersion rather than raising it.** The scheme's spatial and
     temporal truncation errors partially cancel, maximally at the stability limit — the
     dispersion relation is exact along body diagonals at exactly 1/√3 — so at fixed Δx the
     *largest* stable timestep is the most accurate one, and Δt → 0 converges to the
     semi-discrete solution, not the true one. Axis-aligned phase error at 4 kHz falls from
     1.90% to 1.65%, widening the 2%-dispersion bandwidth from 4.1 to 4.4 kHz.

     The listener is sampled once per step, so the output rate drops with the step rate.
     Nothing representable is lost: the grid's own bandwidth is c/2Δx = 17.2 kHz for the
     speaker scenes, far below either Nyquist. References were regenerated at the new rates
     (`gen/cuda/<scene>/<rate>/`, originals kept), so the comparison stays
     same-discretization and all three verdicts are unchanged.

     **Constraint when retiming:** the FDTD and shader batch durations must be equal under
     integer division — `(FdtdSrate/BlendRate)/FdtdSrate == (ShaderSrate/BlendRate)/ShaderSrate`.
     Violating it drifts the source clock against the acoustic clock (66000 Hz gives
     1.00000 ms against the shader's 0.99773 ms, 24 ms of skew over a run) and decorrelates
     the output uniformly across every octave — unlike real dispersion, which is always
     monotonic in frequency. 66150 satisfies it; 66000 does not.

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

## Results

Per-scene verdicts and metrics are in `gen/validation.json`, written by `script/ValidateGolden`
and committed. `ValidateGolden` exits nonzero if any scene falls outside its band or its output
length disagrees with the reference, so the file records the run rather than gating it.

Every scene grades `OK` or `OK(env)`. The bubble scenes sit at exactly the divergence level the
CPU-only control experiment predicts, with energy within ±1.5%.

### Scenes added from the wavesolver dataset

Cymbal and WineglassTap are not in the WaveBlender dataset. They come from the earlier
[wavesolver dataset](https://graphics.stanford.edu/projects/wavesolver/dataset/dataset_table.html)
(Wang et al. 2018). Their thin-shell and rigid-body files are already in the formats our
`Shell` and `Modal` shaders read, so `script/FetchScenes` only renames and moves them
(`animation/`, `shader/`, and `<prefix>.{obj,tet,geo.txt,modes,impulses.txt}`) and converts
nothing. Cymbal is the only scene that exercises the `Shell` path: 88200 frames of per-vertex
displacement and acceleration at 44.1 kHz, 2481 vertices, 10 GB.

The archives contain only the 2018 solver's own config, so ours are in `config/`:

- **Grid and rates from the WaveBlender paper's Table 1** where it lists them (Cymbal: 10 mm,
  80³, 88.2 kHz step rate), and from the wavesolver table otherwise (Wineglass: 5 mm, 1 s,
  120 kHz — the same rates SpollingBowl runs at).
- **Blend rate 980 Hz for Cymbal, not the paper's 1000.** `NSamples = shader_srate / blend_rate + 1`
  truncates, so a blend rate that does not divide both rates exactly drifts the shader clock
  against the FDTD clock. 980 divides 44100 (45 shader samples per batch) and 88200 (90 steps)
  and is the closest such rate to the paper's.
- **Shader rate pinned to 44.1 kHz for Cymbal**, because the shell reader indexes frame files
  as `step + lookahead` and assumes that rate and a zero start time.
- **Listeners moved inside the grid.** Both archives list far-field listening points, and the
  2018 solver propagated to them. Cymbal's is 1.9 m from the object, on a 0.8 m grid. Ours
  samples pressure at a grid cell, so each listener keeps the original direction from the object
  at a radius that clears the 8-cell PML: Cymbal 0.2 m above the disc, Wineglass 3 cm from the
  bowl.
- **Identity vertex map** for Cymbal: the published zip's `misc/` is empty, so there is no
  `vertex_map.txt`, and both solvers fall back to identity.
- **A two-keyframe rest pose** for WineglassTap. The glass never moves and the scene has one
  impulse, but a `Modal` object with no animation file leaves both solvers' pose quaternions
  uninitialized, so the fetch script writes an identity `displace.txt`.

Cymbal's `OK(env)` is amplified CPU drift, not a structural difference. The first differing
sample is #27, at magnitude 1e-19 and a relative difference of 6e-8 — one float32 ulp, in the
first batch, before any fresh cell exists. The moving shell amplifies it from there: cells flip
between solid and air a batch earlier or later, and the error reaches 1.4e-2 by 10 ms. It is not
our per-component fresh-cell solve. `ACOUSTIC_GLOBAL_LSTSQ=1` reproduces the same 8.481e-2 to
every digit and the same first differing sample. Energy matches within 0.11% and correlation is
0.9964.

## Bugs found during validation

- **apply_shader write race** (reference + original port): see above. Detected because
  the port is otherwise bit-reproducible run-to-run and FillerUp wasn't.
- **Upstream FluidSound UB** (`Integrators.h`): `Solver::~Solver()` deletes derived
  integrators through a base pointer with no virtual destructor. gcc compiled it
  benignly; Homebrew clang 22 at -O2 compiles the delete to a trap, which crashed
  GlassPour at teardown and truncated the final batch. Patched in our vendored copy
  (`LOCAL PATCH`); upstream is archived.

## Performance

Per-scene wall clock is in the README. Each mechanism is documented where it lives in the code.

The solver sits at measured hardware limits: the fused FDTD kernel sustains 752–766 GB/s
at 80³ and above, which is this machine's DRAM roof (a pure streaming copy of the same
element count reaches 725–741 GB/s there), and the coupled-bubble scenes are bound by the
matrix units' aggregate factorization throughput — the same 13,889 endpoint inversions cost
154s of worker time on 4 threads and 306s on 8, for the same wall clock. Those two account
for ~95% of the suite.

### Instruments and toggles

- `ACOUSTIC_PROFILE=1`: per-phase wall time at exit — `gpu/exec` (total GPU busy) and its
  `gpu/exec_fdtd` / `gpu/exec_prologue` split, `gpu/idle` (gaps between command buffers on
  the GPU timeline, the pipeline-bubble measure) and its `gpu/idle_fdtd` /
  `gpu/idle_prologue` split, `cpu/sync_window`, `gpu/wait`, `startup/scene_load`, the
  coupled-bubble main-thread breakdown (`bubble/rk4` and, within it, `bubble/apply` — the two
  endpoint matrix-vector products, ~85% of the bubble scenes' CPU time — plus
  `bubble/refactor`, `bubble/fetch`, `bubble/unpack`, `bubble/pack_data`, `bubble/pack_vels`,
  `bubble/read_mesh`), and the `[apply-path]`, `[shader-faces]` and `[bubble-pipeline]` lines,
  the last of which also splits worker time into mass fill, factor, symmetrize and derive.
  Scopes on per-timestep paths resolve their accumulator once (`profile::Phase`) — the map
  lookup otherwise costs more than the phase, and inflated a bubble scene 40%.
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
script/FetchScenes                 # download the scene data into Scenes/ (~12 GB, once)
script/FetchScenes --verify        # checksum Scenes/ against script/scenes.sha256
script/Build                       # build (Homebrew LLVM, C++23)
script/ValidateGolden --profile    # render every scene once: verdicts + wall + phase timing
script/ValidateGolden --score      # re-score build/*.bin — no rendering
script/ValidateGolden GlassPour    # one scene (either mode)
ACOUSTIC_GLOBAL_LSTSQ=1 ...        # reference-bit-identical fresh-cell solve
script/generate_golden_outputs.sh root@<pod-ip> <port>   # regenerate the reference (CUDA host)
```

**Render each scene at most once per session.** Scoring is a pure function of the listener
bytes and the reference bytes, so any run leaves what a later `--score` needs, persisted in
`gen/validation.json`. And outputs byte-identical to a build whose metrics are recorded
here have identical metrics by construction — `cmp` settles that, re-rendering does not.
