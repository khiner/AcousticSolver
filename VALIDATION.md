# Validation

The Metal port is validated against listener outputs from the reference CUDA implementation (RTX 4090, CUDA 12.8, gcc, Linux — `script/generate_golden_outputs.sh`, committed under `gen/cuda/`).
`script/ValidateGolden` runs every scene and compares the listener pressure output sample-for-sample.
`gen/wav/{cuda,metal}/<scene>/<rate>.wav` holds paired renderings for listening, filed by rate like the references, so a retimed scene's previous rendering stays listenable beside its new one.

The reference outputs are committed, so scoring works from a fresh clone without a CUDA host.
They are filed by the rate they were generated at, so a retimed scene compares against a reference at the same rate.
The scene data is not committed (~12 GB) — run `script/FetchScenes` first.
`Scenes/` is fetched and checksummed, so anything we change about a scene lives in `config/<scene>.json` rather than being edited in place there.

## Key numerics decisions

- All Metal kernels compile with **fast-math disabled**, matching nvcc's default IEEE behavior — this is what makes bit-exact GPU comparison possible.
- Every floating-point expression matches the CUDA reference operation-for-operation.
  Changes are gated on byte-identical listener outputs against the previous build, plus the unchanged bit-exact prefix against CUDA below.
- Six deliberate output-observable changes, all validated:
  1. **Fresh-cell least squares per connected component.**
     The reference's global minimum-norm system is block-diagonal across connected components, so solving per component is identical in exact arithmetic and differs only in float rounding, while being asymptotically cheaper and parallel.
     `ACOUSTIC_GLOBAL_LSTSQ=1` restores the reference-bit-identical global solve.
  2. **apply_shader write-race fix.**
     In the reference, a velocity face claimed by both a velocity-blend shader and a point-source force (5,265 collisions in FillerUp) is raced by two threads, making output **nondeterministic run-to-run** (reproduced on both the CUDA structure and the original port).
     Blend and force points now dispatch as two ordered ranges; non-colliding faces are bit-unaffected.
     The fix moved FillerUp from envelope-level to full waveform agreement with the reference (rel L2 3.4e-2 → 2.0e-4) and tightened HandShake (6.5e-2 → 2.5e-2).
  3. **Explicit bubble mass-matrix inverses.**
     The coupled-bubble solve applies precomputed explicit inverses (`spotrf`/`spotri`, chained by SPD Schur updates — see Performance) instead of factor-and-substitute.
     Same linear systems, different operation order: measured in double, before decision 6 narrowed the path, bubble-scene outputs differed at rel L2 ~1e-6, five orders below the CUDA-reference comparison level (byte-identical for GlassPour, below float32 output quantization).
     A derived inverse is always exactly one update from a fresh factorization, so rounding error never compounds.
     The inline fallback path factors and substitutes exactly as the reference does.
  4. **FDTD timestep raised toward the CFL limit** on TalkFan, Trumpet and 2016Pour.
     The paper sets the step rate from the audio rate rather than from the grid — *"our FDTD timestep size is often higher (88.2 kHz and above) than our target grid resolutions—as determined by the CFL condition"* (§6.1) — leaving those three at Courant numbers of 0.357–0.389 against the 7-point scheme's limit of 1/√3 = 0.577.
     They now run at 0.519–0.536: 1.34× fewer steps for the speaker scenes, 1.5× for 2016Pour.

     **This lowers numerical dispersion rather than raising it.**
     The scheme's spatial and temporal truncation errors partially cancel, maximally at the stability limit — the dispersion relation is exact along body diagonals at exactly 1/√3 — so at fixed Δx the *largest* stable timestep is the most accurate one, and Δt → 0 converges to the semi-discrete solution, not the true one.
     Axis-aligned phase error at 4 kHz falls from 1.90% to 1.65%, widening the 2%-dispersion bandwidth from 4.1 to 4.4 kHz.

     The listener is sampled once per step, so the output rate drops with the step rate.
     Nothing representable is lost: the grid's own bandwidth is c/2Δx = 17.2 kHz for the speaker scenes, far below either Nyquist.
     References were regenerated at the new rates (`gen/cuda/<scene>/<rate>/`, originals kept), so the comparison stays same-discretization and all three verdicts are unchanged.

     **Constraint when retiming:** the FDTD and shader batch durations must be equal under integer division — `(FdtdSrate/BlendRate)/FdtdSrate == (ShaderSrate/BlendRate)/ShaderSrate`.
     Violating it drifts the source clock against the acoustic clock (66000 Hz gives 1.00000 ms against the shader's 0.99773 ms, 24 ms of skew over a run) and decorrelates the output uniformly across every octave — unlike real dispersion, which is always monotonic in frequency. 66150 satisfies it; 66000 does not.
  5. **Modal oscillators no longer advance across a batch boundary twice.**
     A batch's last shader sample and the next batch's first are the same instant — that is what `NSamples = ShaderSrate/BlendRate + 1` means — but the reference steps the oscillators at both, so they ran `NSamples / (NSamples - 1)` fast.
     With WineglassTap's 48 kHz shader rate and 1 kHz blend rate that is 49/48, which sharpens **every modal frequency by 2.08%** and leaves a discontinuity in the modal velocity once per batch.

     Both are measurable against the mode data, which fixes the frequencies exactly (`OmegaSquared` in `<prefix>.modes`, damped by the material's Rayleigh coefficients).
     The wine glass rings at 497, 559, 1975, 2106 Hz; the reference put its peaks at 507, 570, 2017, 2150 — each within 1 Hz of the mode times 49/48 — and carried a comb at multiples of the 1 kHz blend rate, 27 dB below the peak, from the per-batch discontinuity.
     After the fix the peaks land on the modes and the comb falls to -124 dB.
     The independent SonicRadiation solver, which has no blend rate and no shader resampling, put its peaks on the analytic modes all along; that is what surfaced this.

     The committed references were generated by the reference implementation and carry the bug, so the three Modal scenes (LegoDrop, SpollingBowl, WineglassTap) now agree with them at envelope level rather than waveform level.
     Restoring waveform agreement needs the references regenerated on a CUDA host with the same one-line change.
  6. **The bubble inverse path runs in single precision.**
     Mass construction, the endpoint factorization and inversion, the chained Schur derivation, and the apply products all use `float`.
     These matrix units turn `spotrf`+`spotri` 2.6–3.1× faster than the double pair at N = 955–1800, and `sgemv` 2.4–4.1× faster than `dgemv`, which the apply is bound by reading.
     PaddleSplash 51.9s → 26.3s, GlassPour 5.5s → 4.5s, measured back to back from builds at the commit and its parent.
     The oscillator state, the RK4 integration and the inline Cholesky fallback stay double, and a matrix that is not positive definite in single precision falls back to that path, so this can cost time but never correctness.

     Against the previous double output: PaddleSplash corr 0.999924, rel L2 1.23e-2, peak difference −27.7 dB — GlassPour corr 0.999987, rel L2 5.1e-3, −28.0 dB.
     That is 38× (PaddleSplash) to 72× (GlassPour) smaller than the platform divergence below, so all three bubble verdicts and their golden metrics are unchanged and no references needed regenerating.
     Scoring both against the reference is *not* sufficient to establish this: these scenes are chaotic enough that two runs can each sit at corr ≈ 0.89 against CUDA while differing from each other, so the two builds were rendered and diffed directly.

## What "matching" means across platforms

Three regimes, in decreasing strictness:

1. **Bit-exact.**
   CupPhone matches CUDA bit-for-bit for 4.66 of 8 simulated seconds — until the first moving-geometry batch — including through the fused-kernel pipeline.
   TalkFan and Trumpet match to rel L2 ~1e-4..1e-5 (−73 to −95 dB) over 10+ seconds with animated geometry.
2. **Ulp-accumulation drift.**
   CPU-side per-batch math (quaternion slerp, fresh-cell least squares, modal ODE stepping) picks up last-ulp ARM-vs-x86 differences from libm and Eigen reduction orders, accumulating as phase drift: LegoDrop −59 dB, SpollingBowl −44 dB, correlation ≥ 0.99998, matching spectra.
3. **Phase decorrelation (chaotic CPU paths).**
   The coupled-bubble ODE amplifies ulp differences to macroscopic divergence.
   **Control experiment**: the CPU-only FluidSound solver alone, ARM/clang vs x86/gcc on identical GlassPour input, diverges at rel L2 = 0.40, corr = 0.92 — the same magnitude as the full Metal-vs-CUDA comparison (0.37, 0.93).
   Envelope (RMS within ±6%) and magnitude spectrograms remain matched, so those are the meaningful invariants, graded `OK(env)`.
   For scale, the deliberate single-precision change in decision 6 moves the same outputs by rel L2 5.1e-3–1.23e-2, 38–72× below this floor.

A structural port bug would fail all three gates at once (wrong energy, wrong spectrum, zero bit-exact prefix); nothing does.

## Results

Per-scene verdicts and metrics are in `gen/validation.json`, written by `script/ValidateGolden` and committed.
`ValidateGolden` exits nonzero if any scene falls outside its band or its output length disagrees with the reference, so the file records the run rather than gating it.

Every scene grades `OK` or `OK(env)`.
The bubble scenes sit at exactly the divergence level the CPU-only control experiment predicts, with energy within ±1.5%.

### Scenes added from the wavesolver dataset

Cymbal and WineglassTap are not in the WaveBlender dataset.
They come from the earlier [wavesolver dataset](https://graphics.stanford.edu/projects/wavesolver/dataset/dataset_table.html) (Wang et al. 2018).
Their thin-shell and rigid-body files are already in the formats our `Shell` and `Modal` shaders read, so `script/FetchScenes` only renames and moves them (`animation/`, `shader/`, and `<prefix>.{obj,tet,geo.txt,modes,impulses.txt}`) and converts nothing.
Cymbal is the only scene that exercises the `Shell` path: 88200 frames of per-vertex displacement and acceleration at 44.1 kHz, 2481 vertices, 10 GB.

The archives contain only the 2018 solver's own config, so ours are in `config/`:

- **Grid and rates from the WaveBlender paper's Table 1** where it lists them (Cymbal: 10 mm, 80³, 88.2 kHz step rate), and from the wavesolver table otherwise (Wineglass: 5 mm, 1 s, 120 kHz — the same rates SpollingBowl runs at).
- **Blend rate 980 Hz for Cymbal, not the paper's 1000.**
  `NSamples = shader_srate / blend_rate + 1` truncates, so a blend rate that does not divide both rates exactly drifts the shader clock against the FDTD clock. 980 divides 44100 (45 shader samples per batch) and 88200 (90 steps) and is the closest such rate to the paper's.
- **Shader rate pinned to 44.1 kHz for Cymbal**, because the shell reader indexes frame files as `step + lookahead` and assumes that rate and a zero start time.
- **Listeners moved inside the grid.**
  Both archives list far-field listening points, and the 2018 solver propagated to them.
  Cymbal's is 1.9 m from the object, on a 0.8 m grid.
  Ours samples pressure at a grid cell, so each listener keeps the original direction from the object at a radius that clears the 8-cell PML: Cymbal 0.2 m above the disc, Wineglass 3 cm from the bowl.
- **Identity vertex map** for Cymbal: the published zip's `misc/` is empty, so there is no `vertex_map.txt`, and both solvers fall back to identity.
- **A two-keyframe rest pose** for WineglassTap.
  The glass never moves and the scene has one impulse, but a `Modal` object with no animation file leaves both solvers' pose quaternions uninitialized, so the fetch script writes an identity `displace.txt`.

Cymbal's `OK(env)` is amplified CPU drift, not a structural difference.
The first differing sample is #27, at magnitude 1e-19 and a relative difference of 6e-8 — one float32 ulp, in the first batch, before any fresh cell exists.
The moving shell amplifies it from there: cells flip between solid and air a batch earlier or later, and the error reaches 1.4e-2 by 10 ms.
It is not our per-component fresh-cell solve.
`ACOUSTIC_GLOBAL_LSTSQ=1` reproduces the same 8.481e-2 to every digit and the same first differing sample.
Energy matches within 0.11% and correlation is 0.9964.

## Bugs found during validation

- **apply_shader write race** (reference + original port): see above.
  Detected because the port is otherwise bit-reproducible run-to-run and FillerUp wasn't.
- **Upstream FluidSound UB** (`Integrators.h`): `Solver::~Solver()` deletes derived integrators through a base pointer with no virtual destructor. gcc compiled it benignly; Homebrew clang 22 at -O2 compiles the delete to a trap, which crashed GlassPour at teardown and truncated the final batch.
  Patched in our vendored copy (`LOCAL PATCH`); upstream is archived.

## Performance

Per-scene wall clock is in the README.
Each mechanism is documented where it lives in the code.

The solver sits at measured hardware limits: the fused FDTD kernel sustains 752–766 GB/s at 80³ and above, which is this machine's DRAM roof (a pure streaming copy of the same element count reaches 725–741 GB/s there), and the coupled-bubble scenes are bound by the matrix units' aggregate factorization throughput — the same 13,889 endpoint inversions cost 154s of worker time on 4 threads and 306s on 8, for the same wall clock.
Those two account for ~95% of the suite.

### Instruments and toggles

- `ACOUSTIC_PROFILE=1`: per-phase wall time at exit — `gpu/exec` (total GPU busy) and its `gpu/exec_fdtd` / `gpu/exec_prologue` split, `gpu/idle` (gaps between command buffers on the GPU timeline, the pipeline-bubble measure) and its `gpu/idle_fdtd` / `gpu/idle_prologue` split, `cpu/sync_window`, `gpu/wait`, `startup/scene_load`, the coupled-bubble main-thread breakdown (`bubble/rk4` and, within it, `bubble/apply` — the two endpoint matrix-vector products, ~85% of the bubble scenes' CPU time — plus `bubble/refactor`, `bubble/fetch`, `bubble/unpack`, `bubble/pack_data`, `bubble/pack_vels`, `bubble/read_mesh`), and the `[apply-path]`, `[shader-faces]` and `[bubble-pipeline]` lines, the last of which also splits worker time into mass fill, factor, symmetrize and derive.
  Scopes on per-timestep paths resolve their accumulator once (`profile::Phase`) — the map lookup otherwise costs more than the phase, and inflated a bubble scene 40%.
- `ACOUSTIC_APPLY_PATH=plain|folded`: pin the boundary-application path.
- `ACOUSTIC_GLOBAL_LSTSQ=1`: reference-bit-identical global fresh-cell solve.
- `ACOUSTIC_LAZY_COMMIT=1`: commit prologue/FDTD buffers at the sync instead of eagerly.
- `ACOUSTIC_BUBBLE_WORKERS=<n>`: bubble inverse-pipeline worker count.

### Benchmarking method

- Runs are bit-reproducible, so `cmp` of listener `.bin` outputs is the exactness gate.
- Time only on a quiet machine, and mind that thermal drift dominates cross-run GPU timing here: a GPU-bound scene measures 43s cool and 53s heat-soaked.
  Start every run from an idle GPU and compare against recorded cool numbers, or measure inside one run.
- Kernels compile at runtime from `src/FDTD/`, so a saved binary is not a snapshot without its matching MSL sources.

## Rerunning

```
script/FetchScenes                 # download the scene data into Scenes/ (~12 GB, once)
script/FetchScenes --verify        # checksum Scenes/ against script/scenes.sha256
script/Build                       # build (Homebrew LLVM, C++23)
script/ValidateGolden --profile    # render every scene once: verdicts + wall + phase timing
script/ValidateGolden --score      # re-score build/*.bin — no rendering
script/ValidateGolden GlassPour    # one scene (either mode)
script/ValidateRadiation           # the SonicRadiation gate: ladder + scene renders (~8s)
ACOUSTIC_GLOBAL_LSTSQ=1 ...        # reference-bit-identical fresh-cell solve
script/generate_golden_outputs.sh root@<pod-ip> <port>   # regenerate the reference (CUDA host)
```

**Render each scene at most once per session.**
Scoring is a pure function of the listener bytes and the reference bytes, so any run leaves what a later `--score` needs, persisted in `gen/validation.json`.
And outputs byte-identical to a build whose metrics are recorded here have identical metrics by construction — `cmp` settles that, re-rendering does not.

# SonicRadiation solver

`src/Radiation/` implements [Jin et al. 2025](https://arxiv.org/abs/2508.08775) from the paper — no reference implementation is public, so there is nothing to compare against sample-for-sample.
Validation is instead against analytic truth, through `build/RadiationTest`.

The harness is a ladder, each rung isolating one layer, run against a unit monopole inside a closed icosphere whose analytic Neumann data drives the solver (the paper's Sec. 5.1 setup: 0.7 m domain, 1 kHz, object 1/6 of the domain, SNR measured on nested boxes at 2.4–4.2× the object).
What the source is, where it sits, and what shape the body is decide how much of the solver a rung reaches, so V2 is gated at six configurations (`--src-offset` in body radii, `--dipole`, `--ellipsoid`, `--box`, `--lshape`) — three sources ordered by what they ask of the surface trace, constant then varying then sign-changing, and three that change the body instead.
At the centre — the paper's setup — the trace is *constant*, which the element-constant basis represents exactly, so that rung prices none of the basis's representation error: what is left in it is quadrature, the time discretization and the grid.

```
build/RadiationTest                    # the default ladder: self-test, V1, V2 (~0.7s)
build/RadiationTest --gate             # every rung checked, nonzero exit on a miss (~3s)
build/RadiationTest --self-test        # CQM weight identities
build/RadiationTest --v1 --full        # pure TDBEM, dense left-matrix solve
build/RadiationTest --v2 --res 60      # the hybrid at a given grid
build/RadiationTest --v2 --check-sets  # verify the set-difference tables mid-run
build/RadiationTest --no-kernels       # the same rungs, weights per pair by transform
build/RadiationTest --stability --time 10
build/RadiationTest --move 2.0 --epoch 16
build/RadiationTest --mesh <obj> --cell <metres>
script/RadiationModes                  # the two WineglassTap mode tables below
```

Every rung prints; `--gate` also checks. It runs each one at the settings the table below was recorded at, ignoring the sweep flags, and exits nonzero if any measure has moved — a threshold belongs to the configuration it was recorded at, and the sweeps exist to leave that configuration.
The margins are for a future toolchain's rounding rather than for drift, since the ladder is deterministic to every digit it prints.

## Results

| rung | measure | value |
|---|---|---|
| self-test | BDF2 stencil, pure delay, DC limits | exact to 3e-15 |
| V1 (pure TDBEM) | surface / radiated SNR | 25.1 / 24.7 dB |
| V2 30³, centred monopole | grid / Eq. 5 SNR | 35.5 / 25.2 dB |
| V2 30³, monopole at 0.5R | grid / Eq. 5 SNR | 29.5 / 23.9 dB |
| V2 30³, centred dipole | grid / Eq. 5 SNR | 20.5 / 18.6 dB |
| V2 30³, ellipsoid body | grid / Eq. 5 SNR | 33.9 / 23.8 dB |
| V2 30³, box body | grid / Eq. 5 SNR | 37.2 / 24.5 dB |
| V2 30³, L-shaped body | grid / Eq. 5 SNR | 32.8 / 23.8 dB |
| V2 60³ | grid / Eq. 5 SNR | 36.7 / 38.4 dB |
| set algebra | max relative mismatch | 0 (exact) |
| stability, 10 s at 30³ | max\|p\|, zero damping | 0.62, no growth over 254,612 steps |

The two output paths have opposite error profiles, which is why both are implemented and written: the Eq. 5 retarded-potential path is free of grid dispersion but accumulates the convolution quadrature's frequency warping, so it converges O(tau²) with the grid (25.3 → 39.1 dB from 30³ to 60³); the grid path is warp-free in steady propagation but carries a floor set by the injection radius, which is a fixed number of *cells* and so barely moves with resolution (35.5 → 36.7 dB).

**Where the source sits is the ladder's sharpest setting, and the paper's is the flattering one.**
A monopole at the centre of a sphere puts the same Neumann data on every element, so the element-to-element coupling the weight tables encode is exercised in the most symmetric configuration it has, and the exact surface trace lies in the discrete trial space.
Displacing it to 0.5R along a direction off every axis costs 2.7 dB of surface trace, 6.0 dB of grid SNR and 1.3 dB of Eq. 5 at 30³.
That is convergence, not a defect — off centre improves with resolution just as the centred rung does (surface 21.8 → 24.9 → 27.2 dB and Eq. 5 24.0 → 28.2 → 32.1 dB at 30³/40³/50³) — but it converges *slower*, so the gap widens rather than closing, which is what a rung whose answer is exactly representable looks like next to one whose answer is not.
A dipole goes further: its trace changes *sign* across the body, and the solver reads 16.6 / 20.5 / 18.6 dB of surface, grid and Eq. 5 at 30³ — another 5.2 / 9.0 / 5.4 dB below the off-centre monopole, and 7.9 / 15.0 / 6.7 below the paper's own rung.
That is the sharpest rung the harness has, and the reason to want it is that the double-layer series is a partly cancelling combination, the window trim thresholds against a window's own peak and the block scales are sized to it, so all three are far more exposed when what they sum to cancels.
It converges too — 16.6 → 19.2 → 21.4 dB of surface trace over 30³/40³/50³ — and its extinction is the best of the three at -29.7 dB, so the representation formula is not what the low numbers are about.
Changing the *body* instead of the source reaches somewhere else again.
A triaxial ellipsoid — an icosphere with its vertices scaled 1 : 0.7 : 0.5, which only shrinks edges so the subdivision picked against the grid still clears its ceiling — needs no new reference at all: a source inside any closed surface has the free-space field outside it, and nothing in that argument uses the sphere.
Under the paper's own centred monopole it costs 6.1 dB of surface trace and 4.3 dB of extinction while the far boxes barely move (grid 35.5 → 33.9), so it stresses the boundary solve and the interior cancellation rather than the propagation — a different failure surface from the dipole, which moved everything.
It converges as well, extinction -17.2 → -19.2 → -21.4 dB over 30³/40³/50³, and carries its own extinction ceiling rather than widening the one the sphere rungs are held to.
What it buys over a sphere is curvature that is not constant and elements that are not all alike, so element areas, aspect ratios and near-set sizes vary over the body.
A box of the same extent goes further still, and differently.
It is generated at the resolution it runs at rather than remeshed towards one — isotropic remeshing moves vertices tangentially, which is what would round off the very edges it exists to provide — with lattice points shared along the seams so the mesh stays edge-manifold, which the gate asserts.
Its signature is the *surface trace*, 17.8 dB at 30³ and converging to only 19.8 by 50³ while its Eq. 5 path runs 24.6 → 34.2 over the same span.
That slow corner is the physics: the pressure gradient is singular at a convex edge, and an element-constant basis converges slowly against it.
Everything else about it is comfortable — extinction -23.8 dB, better than the sphere's, and a grid path at 37.2 dB — so it isolates the boundary trace almost on its own.
Removing a quadrant of that box across its full depth makes an extruded L, which adds a concavity and a reflex edge.
It is the lowest surface trace on the ladder, 15.9 dB at 30³ and only 17.4 by 50³, and the slowest-converging of any rung.
Its own reason for being there is not the reflex edge, whose exterior wedge is *less* singular than a convex one, but that a concavity is the only case where two parts of the surface are close in space while being far apart across it — and the near sets, the R1/R2 band radii and the Eq. 12 set-difference corrections are all built on Euclidean distance, which on a convex body is interchangeable with distance over the surface and here is not.
Both solids are generated from a lattice of cells, a face emitted wherever a solid cell meets an empty one, so the box and the L come out of the same mesher and both are edge-manifold by construction — which the gate asserts rather than assumes.
All six are gated: the centred monopole on a sphere because it is the paper's comparison, the other five because they are what price the solver.
Read accuracy trades against the dipole, or at least the off-centre monopole: a real body's modal surface motion is a sign-changing trace, not a constant one.

The analytic dipole is the monopole differentiated once along its axis, so `DnAmp` is a second derivative and easy to get wrong; it was checked against a central difference of `Amp` over 3,200 point-and-normal pairs, worst relative deviation 9.5e-9.

**The paper's headline SNR is not reproducible from its stated setup.**
It reports ~60 dB at 30³.
Two independent error models — grid dispersion at the Courant limit, and BDF2 convolution-quadrature warping — each bound that configuration to 25–40 dB, and this implementation lands exactly on those models at every resolution tested.
Its convergence behaviour, not the paper's table, is the correctness standard used here.

## The regression gate

`script/ValidateRadiation` is the whole check: the ladder under `--gate`, then scene renders against reference bytes in `gen/radiation/`.
It takes about 8 seconds.

```
script/ValidateRadiation           # ladder, render, score
script/ValidateRadiation --score   # score what is already in build/radiation/
script/ValidateRadiation --exact   # bytes that differ at all are a failure
script/ValidateRadiation --update  # adopt the rendered outputs as the new references
script/RadiationModes --render     # re-render both solvers at 5/7.5/10 mm, then both mode tables
```

The analytic ladder says the solver is right; the reference bytes say it has not changed.
Both are needed, because the ladder is the icosphere and the monopole and touches none of the scene path — the remesh, the modal Neumann source, the epoch scheduler, the listener writes.
`config/WineglassTap_move.json` is what covers that path: 50 ms of the wineglass translating at 1 m/s along the body diagonal and rotating at 2 rad/s about it, which is 3,290 elements, 2,973 steps and 11 geometry epochs in 4.7 s.
It is committed because every config from the dataset carries an identity animation track, which left the epoch path with no reproducible coverage at all.

Which comparison to believe is the change's to say, and both are reported every run.
A reordering — a sort, a parallelization, a relayout — does not change the arithmetic, so bytes are its gate and `--exact` is extremely sharp there.
A deliberate approximation moves the last digits everywhere by construction: the delay kernels sit 1.3e-5 from the transform they replaced and carried element weights 4.9e-6 from recomputing them, which is 98 and 106 dB.
The default gate is an SNR floor of 90 dB.

`script/RadiationBench` is the other half, for changes meant to be faster rather than merely equal: one command, two builds, rounds alternating between them, every number either build printed reported side by side with each side's spread.
The base build is a `git worktree` under `build/`, because the Metal kernels are read at runtime from the *source* directory baked in at compile time and two binaries built from one tree would share one set of kernels.

## Against the WaveBlender path

WineglassTap is the scene both solvers cover, so it is the cross-check.
There is no expectation of sample agreement — different method, different sample rate (118,888 Hz at the grid's Courant limit against 120,000) — so the comparison is spectral.

| mode (Hz) | WaveBlender | radiation, grid path | radiation, Eq. 5 path |
|---|---|---|---|
| 497 | -41 | -47 | -47 |
| 559 | -15 | -19 | -19 |
| 1975 | 0 | 0 | 0 |
| 2106 | -16 | -22 | -22 |
| 4506 | -106 | -103 | -103 |
| 5131 | -57 | -47 | -47 |
| 6416 | -113 | -114 | -124 |
| 9318 | -110 | -97 | -100 |

dB relative to each output's own peak, from `script/RadiationModes`.

Both tables here were assembled by hand until that script existed.
It reproduces them within 2 dB in every cell, which is the measurement behind the claim that narrowing the weight stream cost this comparison nothing — the weights went from float to half to block-scaled bytes and the lag trim from 1e-4 to 3e-3 across those changes, and none of it is visible in the modes that carry the sound.
The three cells the last trim step did move — 4506 and 6416 on the grid path, 9318 on Eq. 5, by 1 to 3 dB — are all 97 dB or more below the peak, where the comparison has no resolution to speak of.

The band each mode is read over is the one thing this is sensitive to, and it wants to be narrow and absolute: ten 1 Hz bins of slack for a peak that lands slightly off its nominal frequency.
A band proportional to frequency was tried and is badly wrong at the top of the range — +/-128 Hz around 6416 Hz reads that mode 32 dB high, catching a louder neighbour instead of the mode.
Two independent solvers agree within 4-12 dB across more than four octaves, on the same dominant mode, including modes 100 dB down, with decay envelopes of the same shape.
The solver's two listener paths correlate at 0.998 with each other.
Overall level differs by about 2.3x, which the boundary discretisations differ enough to explain — grid faces against surface elements.

This comparison is what surfaced the modal timestep bug above: before it was fixed, the FDTD path's peaks sat at none of the analytic mode frequencies.

## Does it hold quality at coarser grids?

That is the method's strategic claim, and the reason to want it: if it stays usable where the FDTD path does not, the coarser grid pays for the boundary integral.
Measured two ways.

Against analytic truth, on the monopole (`--v2 --res N`, SNR in dB):

| res | 15 | 20 | 30 | 45 | 60 |
|---|---|---|---|---|---|
| grid path | 13.6 | 22.2 | 35.5 | 43.2 | 36.7 |
| Eq. 5 path | 12.2 | 18.1 | 25.3 | 32.0 | 39.1 |

The grid path plateaus in the high 30s from res 30 up — its floor is the injection radius, which is a fixed number of *cells* — and falls off a cliff below it.
The Eq. 5 path converges steadily instead (+13.4 dB for a 2x refinement, the O(tau^2) the convolution quadrature predicts), so it wins at fine grids and loses at coarse ones.
Coarsening is the wrong direction for the path that has no grid in it.

On WineglassTap, per-mode level shift when coarsening from 5 mm, rms over the eight modes from 497 Hz to 9.3 kHz:

| | 7.5 mm | 10 mm |
|---|---|---|
| WaveBlender | 8.0 dB | 6.7 dB |
| radiation, grid path | **2.3 dB** | **4.5 dB** |
| radiation, Eq. 5 path | 4.4 dB | 9.4 dB |

Every row was within 1.1 dB of the hand-computed one it replaced when the script was written; the trim to 3e-3 has since moved the grid path's rms 3.4 to 2.3 dB at 7.5 mm, which is the same handful of near-silent modes moving, not the coarsening behaviour changing.

So the claim is directionally right: the grid path drifts about half as much as the FDTD path under the same coarsening.
What it does not do is pay for itself.

| | 5.0 mm | 7.5 mm | 10 mm |
|---|---|---|---|
| radiation | 461 s | 113 s | 47 s |
| WaveBlender | 2.7 s | 1.5 s | 1.3 s |
| elements | 13,852 | 5,964 | 3,290 |
| tables | 1027 MB | 424 MB | 227 MB |

The element counts and table sizes are current; the render times predate the trim to 3e-3 and are deliberately not re-measured, because they are single uncontended runs rather than the interleaved comparisons the perf section uses — they are read as what a render costs, not as one build against another.
Re-running them says 442 / 121 / 48 s, which has the 7.5 mm case *slower* than the 113 s recorded here on a build that is 2% faster, so the old figures are kept and the trim's cost is quoted from the kernels instead.
They were 713 / 186 / 78 s before the weight stream was narrowed.

Cost falls at roughly h^-3 (elements and tables as h^-2, steps as h^-1), a measured 9.8x for a 2x cell size, which is steep — but at the *same* resolution the radiation solver is 171x slower, and the monopole sweep says its own grid path is unusable below about 15 cells per wavelength.
There is no resolution at which coarsening closes that gap.

Reproducing the scene half: `config/WineglassTap_7.5mm.json` and `WineglassTap_10mm.json` hold the *interior* physical domain fixed at 0.12 m half-extent so only the resolution changes, and step each grid at its own Courant limit.
Run each through both solvers (`--radiation` and not) and compare per-mode levels against the mode data.
Note that `FDTD_srate` must round *up* to the limit, `c sqrt(3) / dx`: rounding down by a quarter of a percent takes the FDTD path unstable and it renders NaN.

The caveats are real: one scene, one listener, eight modes, and the scene half measures drift from each solver's own 5 mm render rather than absolute accuracy.
But the conclusion is not marginal enough for those to overturn it.

What the method is actually worth here is qualitative, not throughput: no rasterization and so no blend-rate imprint, no ghost cells or fresh-cell extrapolation, no host round trip in the time loop, and an exact near field for features thinner than a cell — a 1 mm glass wall on a 5 mm grid.
The bug above is what that independence bought.

## Departures from the paper, and why

- **Near-set radii and a feedback filter.**
  With the paper's R1 (its 3×3×3 cell neighbourhood), the boundary–grid feedback loop — phi → corrections → grid → interpolation → phi — has gain slightly above one: from rounding seed to 7e18 by 100 ms at 30³.
  Pure TDBEM is stable, so the loop is the source.
  CFL margin does not help and air damping only slows it.
  R1=2 plus a one-zero filter ((1 + z⁻¹)/2) on the interpolated far-field feedback nulls the grid-Nyquist mode the loop amplifies, costs no measurable accuracy, and holds max\|p\| at signal level over 254,612 steps with zero damping.
  R1=2 is also worth ~10 dB of grid SNR on its own, because the injection radius is what floors it.
- **The row-diagonal Dirichlet solve is kept.**
  It costs surface-trace accuracy at O(tau) (24.9 dB diagonal vs 42.0 dB with the dense solve at 30³) but barely affects radiated output — smooth trace error radiates poorly.
  `--v1 --full` is the dense reference.
- **Meshes are retargeted to the grid.**
  Element count drives every table near-quadratically, and scene meshes are far finer than the paper's "element no larger than a cell" constraint (WineglassTap's is 102,864 triangles at 0.46 cells max edge).
  `RadiationMesh::Remesh` does incremental isotropic remeshing — split, collapse, valence flips, tangential relaxation with reprojection — to 13,852 elements at exactly 1.05 cells max edge, still closed.
  Collapse alone stalls at 2.7× that count; the flips are what unblock it.
- **Weights are stored as block-scaled signed bytes.**
  A table the GPU alone reads holds one signed byte per weight and a power-of-two scale per lag window, applied once to each window's sum rather than to every weight.
  The two windows must carry separate scales: the double- and single-layer series differ by orders of magnitude and sharing one costs 5-19 dB.
  This is what the convolution passes are bound by, so it is worth 1.52 / 1.56x on the two kernels and takes WineglassTap's tables from 1791 to 1195 MB.
  Accuracy is not the constraint at 8 bits, and the ladder in fact reads *better* here, because the lag-0 double-layer term now leaves the table at full float precision instead of coming back out of a narrowed one.
  (An earlier reading of this said the ladder holds at 6 bits as well. It does not — see the six-bit measurement below.)
  It has to leave before the scale is chosen in any case: it is the largest weight in its window by orders of magnitude, and a scale sized to it would quantize everything behind it to nothing.
  Rows are accumulated in double and narrowed once — single rounding is never worse than double — and a pair carried between geometry epochs dequantizes and requantizes onto the same bytes, so it does not drift.
  That is the epoch path only: a scene's first table set keeps its float weights, because the rest state is seeded by a CPU convolution over them.
- **Lag windows are trimmed at 3e-3** of a series' peak, taking the mean window from 40 lags to 20.
  The knob is worth more than it was now that the convolution passes are bound by their weight stream, and it costs less than it did, since a tail weight that trimming would drop quantizes to nothing against byte weights in any case:

  | trim | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
  |---|---|---|---|---|---|
  | mean window | 26.1 | 24.4 | 22.3 | 20.2 | 17.5 |
  | 30³ grid / Eq. 5 | 35.5 / 25.3 | 35.5 / 25.3 | 35.5 / 25.3 | 35.5 / 25.2 | 35.0 / 24.8 |
  | 60³ grid / Eq. 5 | 36.7 / 39.1 | | 36.7 / 38.9 | 36.7 / 38.4 | 35.0 / 38.1 |

  Stepping to 1e-3 was 15% off the windows for nothing measurable at 30³, on the moving rung, or on the grid path anywhere, and 0.2 dB of Eq. 5 at 60³ where there is most headroom to lose.
  It is worth 1.306 / 1.279 to 1.214 / 1.178 ms/step on the two convolution passes, 3.75 to 3.54 ms on a whole step, and 1147 to 1067 MB of tables.
  What pays for it is the dense left-matrix solve's surface trace, 42.3 to 42.0 dB — the largest single cost anywhere on the ladder, and it falls on the diagnostic reference rather than on the row-diagonal solve that ships.

  **3e-3 is the shipping default**, taken after re-pricing it against the rungs beyond the centred monopole, where it looks better than it does above.
  The table's 0.5 dB of Eq. 5 at 60³ is the centred rung's, which is the one whose other errors are smallest, so the trim stands out in it; the same step costs 0.2 dB off centre, 0.3 on the ellipsoid, 0.3 on the box, 0.2 on the L and 0.1 on the dipole.
  Surface and grid do not move on any rung at either resolution, and at 30³ nothing moves by more than 0.1 dB.
  It is worth 0.971 / 0.962 on the two convolution passes and 0.983 on a whole step, with WineglassTap's tables 1067 to 1027 MB and its mean window 22.9 to 20.8 — half of what 1e-3 itself bought, for a similar ratio of accuracy to speed.
  What it cost on adoption was one threshold beyond the routine tenths, `v1 full` surface at 42.0 to 41.5 dB, which is the same diagnostic rung the 1e-3 step was paid from rather than the row-diagonal solve that ships.
  What it does to a rendered scene is the reason it was taken: over WineglassTap's full second the two listener paths differ by 61.8 and 70.3 dB, and the difference is *modal*, not noise.
  Mode frequencies are unmoved to sub-ppm and decay rates to 0.000 dB; what changes is one quiet mode's amplitude, 0.039 dB on the 559 Hz, against 0.002 dB on the dominant 1975 Hz.
  That is also why the error concentrates in the ring-out — local SNR runs 61.8 dB over the strike to 45.0 dB in the last third, converging on the 559 Hz mode's own -47 dB perturbation as the loud mode dies away.
  So a modal body rings at the same pitches with the same decay, half a percent differently in one overtone.

## Dynamic geometry

Scenes follow their object's animation track.
The tables are rebuilt at geometry epochs whose cadence is set by travel rather than by a step count — an epoch runs until the body has moved 0.25 cells from the pose its tables were built at — and each set is built for the *midpoint* of the interval it serves, which halves the worst staleness for nothing.
Epochs land on batch boundaries so that a batch's Neumann samples all belong to one pose, and the next epoch's tables are built on a worker thread while the GPU steps the current one, so a rebuild costs wall clock only when it outruns the epoch it hides behind.
The run reports that overrun as `stalled on table builds`.
The grid and the history rings are sized over the whole trajectory, since both are fixed for the run while the tables are rebuilt against them; an epoch that would outgrow the rings fails rather than reading a wrapped slot.

Carrying an epoch's element weights over rather than recomputing them is a third off that table, and holding the builder's rows at the width the table ends in takes another 16% off the phase that does it — 0.0353 to 0.0298 s on the gate scene, 0.133 to 0.125 s over a whole epoch build, four interleaved rounds with the first epoch's phases as an unchanged control.
It buys much less memory than it looks like it should: a moving run's peak is the epoch alive beside the one the GPU is bound to, and the rows are a small enough part of that to move peak RSS from 3.63 to 3.60 GB.

WineglassTap translating and rotating 0.2 rad/s, 50 ms at 5 mm, gives 10 epochs of 595 steps at 1 m/s and 25 of 238 at 4 m/s.
What an epoch costs is measurable and steady — 0.78s to build, 0.070s to bind — but what a moving *render* costs is not, on this machine: the run time turns on a race between the worker building the next epoch and the GPU stepping the current one, and that race is exactly what other load perturbs.
Stall on the same 4 m/s scene read 0.00, 1.14, 2.57, 3.93, 5.00 and 23.38s across six runs.
Treat the per-epoch costs as the measured quantities and the speed at which they stop hiding as unmeasured.

The listener output stays within 5% of the static render's level with no step at any epoch boundary — the largest sample-to-sample jump anywhere near one is 2.5x the median, against 6.7x for the loudest transient in either render.

`--move` translates the body in the monopole harness, rebuilding the geometry tables every `--epoch` steps.
What governs accuracy is the travel per epoch, not the speed: at 2 m/s the grid path holds 34.6 dB at 0.054 cells/epoch and falls to 29.1 dB at 0.86.
The Eq. 5 path is insensitive to the cadence entirely (25.6 dB throughout), because it reads the boundary histories with no grid state to reconcile.

The paper does not address what a near-set change owes the stored far field.
`--rebase` implements the exact correction — an element leaving a cell's near set has its retarded potential added back into p(F_c), one entering has it removed, at both time levels, evaluated with the old pose's weights.
It measures as a wash: 35.1 vs 35.2 dB at 0.5 m/s, 34.2 vs 34.6 at 2 m/s.
The grid carries an FDTD-transported approximation of the far field, so substituting an exact retarded potential for a migrating element trades one inconsistency for another of the same size.
It ships opt-in, as the reproducible measurement behind that conclusion.

## Performance

`ACOUSTIC_RAD_KERNEL_TIMES=1` gives per-kernel GPU busy time (each dispatch in its own command buffer, so it costs wall time — a diagnostic, not a mode to measure in).

Rebinding a geometry epoch copies every table into its GPU buffer on the thread feeding the GPU, and that copy is spread over the machine rather than run on one core.
On the moving gate scene it takes the steady-state upload from 0.0049 to 0.0030 s an epoch, 41 to 67 GB/s over 201 MB.
It is safe unsynchronized because the GPU is idle there: the batch before a swap is read back through `GridSamples()`, which syncs, before the swap is reached.
What it does *not* touch is the handful of epochs that cost three to six times the rest — those are buffer growth and first-touch faults on fresh pages, identical serial or parallel, and they are once per allocation rather than once per epoch.
Binding is 88% upload and 4% of that render, so this is worth about half a percent of it; eliminating the copy outright, by having the worker build into a second set of buffers, would buy the remaining 0.003 s an epoch for double the table memory, which is not a trade worth making.
The 0.13-0.19 s an epoch this was recorded at does not reproduce — it is 0.008 s here, on 201 MB of tables.

`ACOUSTIC_RAD_FLOOR_PROBE=1,2,3,4,5` attributes what the two convolution passes still cost with their entry loop capped to one entry per source — the floor left over when almost no convolution is done.
Each level is a cut-down build of both passes (see `RadiationParams.h` for what each removes: 1 is the floor itself, 2 drops the ring traffic behind the staged history, 3 the `QuadSum` reduction, 4 the scatter writes, 5 everything but the dispatch).
They are compiled as function constants, so the shipping path carries no branch, and every requested level is dispatched *beside* the real passes each step and timed on its own slot — one process, one step, one thermal state, which is the only way these legs are comparable.
They write to scratch of the same shape as the real outputs, so a probe run stays bit-correct and `script/ValidateRadiation --exact` passes underneath it; what it costs is the extra dispatches and ~90 MB.
Read the deltas, not the absolutes, and read them as what removing each one buys rather than as a partition of the floor: they total more than it, because taking any one out lets the rest overlap better.

Measured on WineglassTap at 5 mm, five rounds, the floor is **0.148 / 0.104 ms/step** on the cell and element passes — 12.2% and 8.6% of them, and 6.6% of a whole step.

| removing | cells | elems |
|---|---|---|
| ring traffic behind the staged history | 0.049 | 0.024 |
| `QuadSum` reduction | 0.035 | 0.010 |
| the write block | 0.054 | 0.015 |
| dispatch alone | 0.035 | 0.035 |

The one item worth naming is the last: a threadgroup per source element costs 0.035 ms/step on *both* passes before it does anything at all, a quarter of the cell floor and a third of the element floor, for 13,852 threadgroups of 256 threads that return immediately.
Mapping several source elements to a threadgroup is the change that would address it, and it is worth 1.8% of a step at most.
Nothing else in the floor is worth more than 1.4% of a step, so the floor is understood rather than opened.
Two corrections to what was written here before the measurement: the write block is the largest single item on the cell pass, not the negligible one arithmetic predicted, and level 4 prices that whole block — it drops the loads of `dst`, `xs` and `d0` with the two stores — so the stores alone are not what it measures.

WineglassTap (64×66×64 grid grown to clear the shell, 13,852 elements, 5.15M cell-table and 7.51M element-table entries, 30.5M interpolation references), on an M5 Max:

| | ms/step | tables |
|---|---|---|
| first working port | 18.10 | 3227 MB |
| source-major tables, staged history, completed convolutions | 14.64 | |
| parallel Eq. 5 listener pass | 11.46 | |
| compact table records | 11.30 | 2901 MB |
| half-precision weights | 9.86 | 1697 MB |
| branch-free split of the two kernels' inner loops | 6.51 | 1697 MB |
| four-lag-aligned tables, vectorized convolution, results stored by target | 5.90 | 1791 MB |
| block-scaled signed byte weights | 4.4 | 1195 MB |
| twelve-byte staged records | 4.3 | 1147 MB |
| lag windows trimmed at 1e-3 | 3.88 | 1067 MB |
| lag windows trimmed at 3e-3 | **3.81** | 1027 MB |

The first six rows are short-run figures on an idle GPU, for comparison against each other.
The last one is not comparable to them arithmetically: this machine's GPU drifts far enough with heat to swamp the difference, so it was measured interleaved against the build above it in the same sweep, three rounds each — 7.18 / 7.23 / 7.28 ms/step against 5.81 / 5.90 / 6.00.
The gain is that 1.23x, on a run where the older build reads 7.23 rather than its own cool 6.51.
The full 1 s render agrees: 118,888 steps in 706 s, 5.94 ms/step, against the 875 s and 7.53 ms/step the previous build took over the same scene.
The two rows after it are interleaved ratios rather than absolute figures, scaled onto the 5.94 above: block-scaled bytes measured 5.29 against 3.935 ms/step over a 0.05 s render, base spread 0.02, and the narrowed records 1.367 / 1.343 against 1.319 / 1.285 on the two kernels.
The 1e-3 row is measured outright — the full 1 s render, 118,888 steps in 461.3 s, which is 7.7 minutes against the twelve this scene took at the top of the table.
The 3e-3 row is an interleaved ratio again, 0.983 of the row above it over two runs with every control kernel within 0.6%, and is *not* a measured render.
Two full renders of it were taken and they say the opposite — 412.6 s at 1e-3 against 442.2 s at 3e-3 — because they ran back to back and the second one ran hot.
That is the clearest example this file has of why whole-render wall clock is not comparable across runs on this machine, and it is kept here as one.

Every step of that is accuracy-neutral except the last, which spends 0.039 dB on one mode's amplitude — see the lag-trim entry above.

Where the time goes now, per-kernel GPU busy time on the same scene:

| pass | ms/step |
|---|---|
| element-element convolution | 2.06 |
| cell-element convolution | 2.02 |
| element Dirichlet update (Eqs. 17-20) | 0.80 |
| set-difference corrections (Eq. 12) | 0.18 |
| PML transport and both listener paths | 0.12 |

What the measurements said, in case it saves the next attempt:

- **The convolution passes are the solver.**
  They are ~70% of a step.
  Everything else — the grid transport, the PML, the corrections, both listener paths, the element update — is the remaining 30%, and the scalar FDTD interior itself is 0.4%.
- **What bounds them is the weight stream**, and the probes establishing that are in the section below.
  The rest of this list is what was measured *before* the inner loops were optimized to the point where bytes took over, and the byte-weight row of it has since been reversed and shipped.
  It is kept as the conditions its neighbours were measured under.
- **What bounded them at the time was work per weight, not weight bytes.**
  Removing 326 MB/step of record traffic changed nothing measurable, while halving the *instruction* count in the inner loops (splitting the double- and single-layer sums so no lane computes both addressings) took them from 6.2 to 4.6 ms.
  And halving the *bytes* — block-scaled signed bytes, one binary exponent per lag window — measured 16-21% **slower** than half then.
  That has since reversed: against the kernels as they now stand the same encoding is 1.52 / 1.56x faster, because the inner-loop work it adds hides entirely under the stalls the smaller stream removes.
  Accuracy was never what stopped it: every SNR on the ladder holds at 8 bits, provided the double- and single-layer windows carry separate scales (sharing one costs 5-19 dB, because the two differ by orders of magnitude).
  Six is a different matter, measured below.
- **Four weights and four staged lags per instruction is worth 13-14%.**
  It needs the windows widened to four-lag boundaries with zero padding, which costs about a tenth more weights — and the record narrows by the source element the staged passes never read, which very nearly pays for it.
- **That padding is cheaper to keep than to reclaim.**
  Storing each window's trimmed lags alone, with the widening carried in the record and the lanes that fall outside a window masked off, takes the tables from 1067 to 1017 MB and leaves the two kernels at 0.993 / 1.028 of what they were.
  The two halves of that separate cleanly.
  Reading the *unchanged* 1067 MB tables through the same addressing costs 1.035 / 1.062: a window no longer begins on a four-byte boundary, so `char4` becomes `packed_char4`, and with ~27-lag windows two quads in seven straddle an end and pay the mask.
  So the 50 MB is worth 3-4% back and the addressing that frees it costs 3.5-6.2%.
  Both are five interleaved rounds on WineglassTap at 5 mm with every control within 0.6%, and both renders were bit-identical — the change is a relayout, so bytes are its gate.
- **Six-bit weights are fast and not free.**
  Four weights in three bytes is 25% off the stream, and it measures 0.883 / 0.947 on the two kernels and 0.955 on the step — *net* of the unaligned window starts it forces and the unpacking, which is the largest speed lever left in these passes.
  What it costs is two bits of magnitude, and the ladder does not hold at six: v2 grid 35.5 -> 34.9 dB, the 3.0x / 3.6x / 4.2x boxes 32.8 -> 32.1, 39.9 -> 37.1 and 33.6 -> 32.8, the moving scene 35.1 -> 34.6, and `--gate` fails on five rungs.
  The Eq. 5 listener path barely notices, at 0.1-0.3 dB — it is the grid path that pays.
  The encoding is not what costs it: the pack and unpack round-trip exactly over all 16.7M quads, host and device agreeing, so this is quantization alone.
  A second exponent per window fits the record's two spare bytes and was built and measured: it recovers the rendered output — the moving scene goes from 42.5 / 44.0 dB against the references to 51.6 / 51.2, three times less error — and takes the ladder loss to 0.4-0.7 dB, but it still fails the same five rungs.
  It does not fix what the rungs are sensitive to.
  A second scale fixes *dynamic range*, the tail of a decaying window crushed by a scale sized to its head, and the rungs are limited by *resolution*: even the leading half holds its weights in five magnitude bits against a byte's seven, and the dominant weights are quantized as coarsely either way.
  No number of scales reaches that floor — only more bits per weight would.
  It costs speed too, and unevenly: 0.887 / 0.998 against the one-scale 0.883 / 0.947, so the per-quad scale select takes the element-element kernel's whole win while the cell kernel keeps its own.
  That is what short windows do — the element-element near pairs run to a few quads, so the halves are one or two quads each and the select is nearly pure overhead, which is also why splitting them would buy little accuracy.
  Splitting the cell table alone therefore looks like the better shape of the two, and has not been built.
- **Thread mapping matters, up to a point.**
  Two, four, eight and sixteen lanes per entry are within 1% of each other, and eight is the best of them; a whole SIMD group per entry costs 30-44% once a lane covers four lags, because a ~27-lag window no longer fills 32 of them.
- **Store results where they are read.**
  The cell convolutions are written by a pass that streams the table by source element and gathered by three that read it by target cell.
  Source-major, a target's references land one source block apart and each costs its own cache line.
  Scattering the 5.2M writes to store them target-major instead cost the writing pass 0.22 ms/step and took the Dirichlet pass from 1.29 to 0.80.
- **A gather pass needs threads.**
  The Eq. 12 corrections ran one thread per band cell — 16,208 threads over 4.2M references, covering none of the gather latency.
  One SIMD group per cell took the pass from 0.60 to 0.18 ms/step.
- **Dispatch shape is worth checking first.**
  The Eq. 5 listener pass was one threadgroup per listener, so with one listener it used about 1/40th of the GPU and cost 3.32 ms/step.
  Splitting each point's entries across threadgroups took it to 0.06.
- **Staging is not what the convolution passes' fixed cost is.**
  They copy each source element's history into threadgroup memory before reading it, and the ring is sized by the listener table, which reaches much further than either convolution table does — on WineglassTap the passes need 40 lags of a 128-lag ring.
  Staging only those 40, which also takes the threadgroup buffer from 1056 to 352 bytes, is worth about 1% against a spread of 3%.
  Capping the entry loop to one entry per source leaves 0.31 / 0.22 ms/step, so that floor is real; it is the per-entry record load, the reduction and the writes, not the staging in front of them.
- **Rejected, measured:** tightening the element near set from the R2 cube to the exact union of the element's own interpolation corners cuts the tables ~20% but costs 0.6 dB of grid SNR.
  Sizing the convolution kernels' threadgroup staging dynamically rather than at the worst case changed nothing (occupancy was not the limit).
  Two entries in flight per SIMD group: 1%.

Building the tables is separate work, and a moving scene pays it once per epoch rather than once per run.
WineglassTap's scene build is 1.9s wall against 6.9s before, and the table build inside it — the part an epoch repeats — is 0.75s against 2.9s.
Two things got it there.

The first is that convolution weights no longer go through a contour sampling and transform per pair.
Both Laplace symbols reach the geometry only through the retarded delay, and with `s = gamma(zeta) / tau` that delay is the dimensionless `theta = r / (c tau)`, so the weights of every non-self pair are a scaled sum of two universal functions of theta — `Psi_j(theta)` and `Chi_j(theta)`, tabulated once over theta in 2.8 MB and read by interpolation (see `CqmKernelTable`).
The transform is what that cost: `ComputePairSeries` fell from 34.9s of thread time to 3.4s, and `vvexp`, `vvsincos` and the FFT together from 18.0s to 0.02s.
Every ladder number above is unchanged, and `RadiationTest --no-kernels` runs the same rungs through the per-pair transform to show it — the self-test reports the tabulation's own error, 8.1e-6 of a series' peak, against the half-precision the weights are stored in.

The second is that what remained was mostly serial.
A build ran at ~3.2x on a machine with far more than that, because the weights were the only part of it spread over threads.
Every other phase is now counted first and written second, or scattered with per-worker cursors, so it runs over disjoint ranges in parallel and lands the same bytes a serial sweep would — verified as a bit-identical render, not just matching table sizes, which survive permutations that change the arithmetic.
`PairTableBuilder::Finish` is the one worth naming: it sorted every entry by (source, target) to match the order the convolution passes stream, and since rows are built in ascending source order that sort is a counting sort, which is both cheaper and parallel, with the weight copy after it a parallel gather.

Together the two took the scene's build line from 6.3s to 1.8s, measured old binary against new, round-robin in one run (6.4 / 6.3 / 6.3 against 1.8 / 1.9 / 1.8).
`ACOUSTIC_RAD_BUILD_TIMES=1` prints where the 0.78s of table build inside that goes, and three rounds of it agree to a few milliseconds:

| phase | seconds |
|---|---|
| element-element weights and assembly | 0.318, or 0.215 once an epoch has one to carry over |
| cell-element weights and assembly | 0.218 |
| resolving stencil refs to entries | 0.114 |
| interpolation stencils (Eqs. 19-20) | 0.046 |
| candidate lists | 0.029 |
| row fill and sort | 0.021 |
| correction stencils (Eq. 12) | 0.017 |
| near-list sweep | 0.017 |

An epoch after the first carries its element-element weights over instead of rebuilding them.
Those weights are rigid invariants — the kernels see only inter-element distances, normals and areas — so a pair in both epochs' near sets keeps what it had and only membership turns over, which is a third off that phase (0.320 / 0.321 / 0.316 / 0.324 against 0.216 / 0.215 / 0.213 / 0.214, interleaved).
It costs the previous epoch's weights in memory, 1.6 GB here.
Exact in exact arithmetic: recomputing transforms the mesh in floating point first, so the two renders agree to 4.9e-6 relative rather than bit for bit, against the 1.3e-5 the delay kernels themselves carry.

Applying an epoch is separate again, and unlike the build it runs on the thread feeding the GPU, so it is never hidden.
It costs 0.070s per epoch (0.63 / 0.62 / 0.65s over nine of them).
Most of what it used to be was work with no Metal in it — the half-precision repack, the reference renumbering, the lag-0 split — which the worker that built the tables now does as well (`PackRadiation`), so the swap is uploads.
The rest was the uploads zeroing every table buffer before overwriting it, and draining the GPU stream whenever one grew: they size with slack now and zero only what the upload does not cover.
The builder's own float weights are released once packed, without which two epochs' worth alongside the packed form is enough to put a 36 GB machine into swap.

An epoch after the first also builds its weights straight into the half-precision layout the GPU reads, rather than building them as float and converting.
Nothing on the CPU reads those two tables — the rest state is seeded by a CPU convolution, but only over the first epoch's, which is why that one keeps its float weights — so the float arrays, 1.11 GB of cell weights and about 1.4 GB of element weights for this scene, are never allocated at all.
`Finish` widens the windows to four-lag boundaries and lifts out the lag-0 double-layer term where the separate packing pass used to, and a carried-over pair copies its half-precision window back, restoring the lifted term so the next table can lift its own.
Time is a wash — 0.79s an epoch against 0.80s, since the work moved rather than disappeared — and the render is bit-identical across eight epochs, staged against not.

Peak resident memory is the frontier that change is for, and it is the one this machine measures worst: the same moving scene read between 9.6 and 15.3 GB across runs of identical code.
Take the arithmetic above as the claim and the RSS as directional.
It matters because peak memory, not speed, is what a finer grid or a larger mesh runs out of first.

A caution on all of the above: whole-render wall clock on this machine is worth nothing across runs, and worth little within them once anything else is competing.
The figures here are either per-phase timers inside one build, or old-against-new interleaved round-robin, and the ones that could not be pinned down that way are called out as unmeasured rather than quoted.

**Amortizing the weight stream over several steps is measured and rejected**, which is worth recording because it read as the one accuracy-neutral lever left.
The idea: a step's single-layer convolution depends on no unknown — the Neumann samples for a whole blend batch are uploaded up front — and neither does the double-layer tail beyond the first few lags, so one pass every B steps could read each weight once and emit B outputs.
Its ceiling is what deleting a convolution sum outright buys, which is nothing worth having.
Removing the single-layer sum from both kernels takes them from 2.049 / 2.078 to 1.993 / 2.055 ms/step, and removing the double-layer sum instead to 1.962 / 2.038 — three interleaved rounds each on WineglassTap at 5 mm, with the untouched kernels flat at 0.011.
So the two sums the kernels are named after are under 7% of what they cost, and a scheme that makes those sums cheaper per output, at a few hundred MB of partial sums and a second set of passes, cannot pay.

Where the time is instead: capping the entry loop to one entry per source element — keeping the staging, the barrier and one entry's record, reduction and write — leaves 0.307 / 0.223 ms/step, so it is not launch overhead or history staging either.
**It is the weight stream.**
Pointing every entry at one shared weight run, which leaves the loop trip counts, the staged history reads, the record loads, the reduction and the writes all exactly as they were and collapses only the bytes, takes the two kernels from 2.053 / 2.071 to **0.718 / 0.818 ms/step** — 2.9x and 2.5x, with the Dirichlet pass flat at 0.838 against 0.835 as the control.
The arithmetic agrees: the element table moves about 124 bytes an entry over 7.51M entries, 931 MB/step, which at 2.07 ms is ~450 GB/s against this machine's ~750 GB/s roof.

Deleting a convolution sum barely moved these kernels because an entry's double- and single-layer windows are contiguous — about 104 bytes for a 52-lag pair — so half the arithmetic still pulls the same cache lines.
Halving the footprint rather than removing it — the same probe compressing the span entries are spread over — gives 2.073 / 2.087 to **1.320 / 1.265 ms/step**, 1.57x and 1.64x, with both control kernels dead flat.
So the relationship is close to linear in bytes, and a weight encoding at half the width is worth about 1.4x on a whole step if its unpacking is close to free.
Bytes per entry are the lever, and that reverses the note below about work per weight rather than weight bytes.
**Whatever was true when that was measured, the kernels as they stand are bound by the weight stream.**
Acting on it: block-scaled signed bytes, previously rejected at 16-21% slower, now take the two kernels from 2.068 / 2.090 to **1.363 / 1.344 ms/step** and the tables from 1791 to 1195 MB, with every other kernel flat and the ladder unchanged or better.
That is 1.52 / 1.56x, against the 1.59 / 1.64x the probe predicted — the gap being the record, which grew from 12 to 16 bytes to carry the two exponents.
Narrowing it back closes most of that: a staged record counts its window fields in *quads* of lags rather than lags, which every one of them is a multiple of, so each fits a byte alongside the two byte exponents.
Twelve bytes again, and worth a further 1.367 / 1.343 to 1.319 / 1.285 ms/step with the output bit-identical.

Every other known lever trades accuracy: coarser meshes, harder lag trimming, or the method's actual claim — equal perceptual quality at coarser grids than the FDTD path, which halves the step count and cuts table entries by roughly the fourth power of the cell size.
