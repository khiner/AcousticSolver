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

# Room-acoustics solver

`src/Room/` implements the finite-volume room-acoustics scheme of [Bilbao, Hamilton, Botts & Savioja 2016](https://www.pure.ed.ac.uk/ws/files/22154168/fv_genimp_final_r3.pdf).
Unlike the other two solvers it has a public reference implementation — [PFFDTD](https://github.com/bsxfun/pffdtd), by one of the authors — so it is validated from both sides: an analytic ladder for the physics, and receiver signals compared sample for sample against the reference's own engines on the identical discrete problem.
Identical is meant literally: scenes are that implementation's voxelizer output, converted rather than re-derived, so there is no discretization difference to explain away.

The interior comes in two stencils, chosen per scene: the 7-point Cartesian one and the 13-point one on the face-centred-cubic sublattice.
Both are validated the same way and against their own references, and the two are compared at equal output bandwidth at the end of this section.

A step is three dispatches on either grid — two where a room's walls are all rigid — where the reference engine's per-step body is ten passes over whole grids, or eleven on the FCC grid.
What the passes did is not skipped, it is folded: the halo mirrors are read rather than written (a step that would leave the grid is a step back onto the opposite neighbour, composed one axis at a time — which is exactly what the three mirror passes left behind), the absorbing shell needs the level the interior update overwrote and that is the value the update already has in a register, which node is on the shell and how hard it absorbs come out of the node's position, and the rigid boundary is the interior update with a node's missing neighbours dropped, so a node's adjacency replaces a second streaming of both levels.
How a node reaches that adjacency is where the two grids part.
Six bits ride beside every Cartesian node for seven eighths of a byte each; twelve do not, and a grid-wide array of shorts at two bytes a node costs an FCC grid more than the second pass it would save, which is what an earlier attempt at the fold measured.
So the FCC masks stay packed in boundary order and a node reaches its row by rank: one occupancy word and one running count per block of 32 nodes, a quarter of a byte a node, and the count of occupied nodes below it in its own block.
Every fold is exact: all five scenes render byte-identical across both changes.

```
build/RoomTest                         # the default ladder: modes, energy, tube, balance, golden
build/RoomTest --fcc                   # the same rungs on the 13-point FCC grid
build/RoomTest --gate                  # every rung checked, nonzero exit on a miss (~19s)
build/RoomTest --modes --count 24      # more eigenmodes
build/RoomTest --tube --length 4000    # a longer impedance tube
build/RoomTest --soak --steps 1000000  # a longer soak
build/RoomTest --roofline --scene ../Scenes/RoomChurch/config.json   # both passes against their own plain streams
build/RoomTest --roofline --fcc --cells 1000 --repeat 20              # the same on a bare air grid of any size
build/RoomTest --implicit              # the Smits & Bilbao 2025 implicit scheme: dispersion, a rigid box, and what it costs
build/RoomTest --golden --scene ../Scenes/RoomChurch/config.json --gen ../gen/room/RoomChurch
script/ValidateRoom                    # the regression gate: the ladder, then all five scenes (~55s)
script/RunRoomReference --score        # what the reference engines score against their own fp64 truth
```

## The ladder

| rung | measure | threshold | value |
|---|---|---|---|
| eigenmodes | worst error vs Morse & Ingard over 16 modes | 0.035% | 0.0329% |
| dispersion | worst error vs the 7-point relation's own prediction | 3e-5 | 1.70e-5 |
| energy | sealed rigid box, drift over 100k steps | 2e-6 | 5.74e-7 |
| impedance tube | worst \|R measured − R analytic\|, 100–1800 Hz | 2e-4 | 8.3e-5 |
| impedance tube | window-edge leakage against the peak | 0.06 | 4.4e-2 |
| energy balance | room + branch storage + dissipation, 20k steps | 1e-5 | 4.3e-6 |
| energy balance | share of the energy the wall took | ≥ 90% | 100.000% |
| soak | settled max\|u\| ratio over 500k driven steps | 1.0001 | 1.0000012 |
| golden | shoebox worst receiver vs the fp32 CUDA reference | 35 dB | 42.4 dB |
| eigenmodes, FCC | worst error vs Morse & Ingard over 16 modes | 2.1% | 2.02% |
| dispersion + wall, FCC | worst error vs the FCC relation carrying its staircase term | 6e-4 | 4.64e-4 |
| energy, FCC | sealed FCC box, drift over 100k steps | 2e-6 | 1.19e-6 |
| golden, FCC | FCC shoebox worst receiver vs the fp32 CUDA reference | 34 dB | 48.8 dB |

RoomShoebox is what makes the first two rungs possible: a whole number of cells on every side, pairwise coprime, with the walls half a cell outside the outermost nodes, so the box modes are exact discrete eigenvectors and the analytic frequencies apply with no fitting.
Its room is also exactly sealed — the voxelizer flags both sides of every wall, the exterior decouples, and the absorbing shell never fires — which the mode rung asserts on every run.
RoomShoeboxFcc is the same box on the other grid, sealed the same way, and the closure check reads it in unfolded coordinates so it is the room being checked and not the folded array.

**An FCC box's modes are not exact discrete eigenvectors, and the two per cent in the table is that and not dispersion.**
A wall drops the neighbours pointing through it and puts their weight back on the node's own diagonal, which says each missing neighbour equals the node itself.
On the Cartesian stencil that *is* the half-cell mirror — the missing neighbour's reflection is the node.
An FCC face diagonal reflects onto a point that is not on the sublattice at all, so the substitution is right only where the mode is flat along that wall.
What is left over is diagonal and first-order perturbation theory gives it in closed form: a node on a wall normal to *a* is missing four diagonals whose true values sum to 2(cos β\_b h + cos β\_c h) times its own where the scheme puts 4 Sl2/A2, and each wall carries two faces of cos²(β\_a h/2) over N\_a cells of the mode's energy, halved again when the mode varies along *a*.
That prediction lands within 4.6e-4 of what the box measures across all sixteen modes, which is what keeps the FCC mode rung as sharp an instrument as the Cartesian one instead of a two per cent shrug.
It is also the reason the two schemes are not equally accurate at a room's surfaces even where their interiors are matched — see the comparison below.

The FCC energy rung's box is seeded on the even sublattice alone.
All twelve face diagonals preserve the parity of the index sum, so the two sublattices the folded array interleaves never exchange an edge, and the odd one stays at zero for the whole run while the even one is a genuine FCC room.
Its padding keeps the fold seam outside the box, so nothing about the folded storage enters the conserved quantity.

**The dispersion rung is against the coefficients in force, not the ideal ones.**
The 7-point relation predicted from an exact λ² is wrong by +5.4e-5 at the low modes, three times the error being measured, because `RoomUpdateCoefficients` rounds λ² and the single-precision diagonal shift moves it again.
The test calls the same function the stepper does and predicts from what comes back.
That shift is worth naming on its own: it contributes +3.1e-4 relative at mode (1,0,0), about ten times the actual dispersion error, so any comparison of low-frequency modes that does not account for it is mostly measuring the shift.

The impedance tube reads reflection as the ratio of a run with a two-branch test wall to the same duct left rigid, so the source spectrum, the transit dispersion and the window all divide out.
Its analytic side is the discrete wall's own admittance: branch impedances recovered from the float quads, evaluated at the trapezoid rule's frequency variable *s* = i(2/Ts)tan(ωTs/2), with β from the scheme's dispersion relation including the shift.
Getting that wrong is loud — a stray factor of λ in κ reads as 0.24 rather than 8.3e-5.

Energy is two rungs because a lossy wall breaks the first one.
Sealed and rigid, the discrete energy is conserved and the rung is how far it moves.
With one lossy wall it is no longer conserved, but room energy plus what the branches store plus everything the wall has dissipated is, and that is the balance rung.
It reads the dissipation off the state as the step's own change in the branch variable, so it needs no extra kernel and the solver carries no extra per-step traffic for it.
All of the drift accrues in the first 2k steps, while the field is large, and is then flat.

## Against PFFDTD

Overall / worst-receiver dB, per scene, against all three reference sets.
`cpu_fp64` is the discrete problem's exact answer up to rounding, and the last column is where the reference's *own* single-precision engine sits against it — the floor no fp32 solver gets past:

| scene | vs cpu_fp32 | vs cuda_fp32 (gate) | vs cpu_fp64 | cpu_fp32's own floor |
|---|---|---|---|---|
| RoomShoebox | 315.53 / 314.34 | 43.60 / 42.41 (35) | 41.82 / 40.73 | 41.82 / 40.73 |
| RoomChurch | 96.13 / 90.80 | 53.34 / 52.19 (44) | 49.99 / 48.91 | 49.99 / 48.90 |
| RoomConcertHall | 100.00 / 97.39 | 60.17 / 55.85 (48) | 56.12 / 53.12 | 56.11 / 53.12 |
| RoomShoeboxFcc | 315.14 / 314.40 | 49.48 / 48.80 (34) | 39.37 / 38.76 | 39.37 / 38.76 |
| RoomChurchFcc | 104.23 / 102.92 | 60.27 / 59.28 (44) | 49.96 / 48.60 | 49.96 / 48.59 |

**All five scenes land exactly on PFFDTD's own fp32 floor** — the fp64 column and the floor column agree to the digits printed.
The gates are that floor less five dB, and they hold with 7 to 15 dB of margin.
Every scene scores higher against the CUDA reference than against the fp64 truth, and that is the scheme rather than luck: single precision is not only rounding here, it carries a diagonal shift of 1.19e-7 that fp64 sets to zero, so all three fp32 engines solve the same slightly different problem and agree with each other before they agree with the truth.

## Bit identity, and where it stops

On both shoeboxes the Metal stepper is **bit-identical to the reference's own CPU fp32 engine**: 315 dB on each, max difference 1.14e-13 on a peak of 485 Cartesian and 1.46e-11 on a peak of 3.11e4 FCC, which is the fp64 summation order in the eight-corner receiver mix and not the stepper at all.
That holds under both stencils, which is what says the 13-point interior, its folded storage, its wider adjacency mask and its absorbing shell are all transcribed rather than approximated.

Off that scene it is structurally unreachable, and the reason is the platforms rather than the port.
Two effects, both measured rather than assumed:

- **The Apple GPU flushes fp32 subnormals and the CPU engine does not.** Scaling the source by an exact power of two — an operation that changes nothing a normal float does — moves our own answer by 95 dB.
- **Clang contracts the reference's boundary loop to FMAs.** Rebuilding the reference with `-ffp-contract=off` moves *its own* answer by 92 dB.

Either perturbation alone saturates two builds at ~90 dB after the church's 12,744 steps, which is what makes ~90–100 dB the identity-equivalent score here — the FCC church, at 104 dB over its shorter run, is the same statement.
The church agreement matrix says the rest: ours against cpu_fp32 **96.1 dB**, cpu_fp32 against its own no-FMA build 91.9, against its own flush-to-zero build 93.8, and against cuda_fp32 **53.3**.
We sit closer to the reference than the reference sits to itself under a compiler flag, and 43 dB closer than its two shipped engines sit to each other.
Sample by sample, receiver 0 of the church is bit-identical for 129 steps and then departs by one fp32 ulp.
Bytes are therefore not a cross-engine gate on any scene with lossy walls, and chasing them across that boundary is chasing the two effects above.

## The regression gate

`script/ValidateRoom` is the whole check and takes about 55 seconds cold, dominated by the concert hall.
It runs `RoomTest --gate` — every rung above at the settings its threshold was recorded at — and then renders all five scenes and scores each twice.

```
script/ValidateRoom             # ladder, render, score
script/ValidateRoom RoomChurch  # one scene
script/ValidateRoom --score     # score what is already in build/room/
script/ValidateRoom --exact     # bytes that differ at all are a failure
script/ValidateRoom --update    # adopt the rendered outputs as the new references
```

The two scores answer different questions and both are reported every run.
Against `gen/room/<Scene>/cuda_fp32.bin` it is the physics statement, floored at 35 / 44 / 48 / 34 / 44 dB on the worst receiver.
Against `gen/room/<Scene>/metal_fp32.bin`, this machine's own committed output, it is a change detector: the renders are bit-reproducible run to run, so anything that moves says the arithmetic moved, and `--exact` is the sharp form of it.
The ladder alone is not enough because neither shoebox has a lossy wall, so the churches and the hall are the only reproducible coverage the frequency-dependent boundary has at scale — 150k to 1.45M lossy nodes against the ladder's 224.

Per-scene render facts, on an M5 Max, wall clock including scene load:

| scene | nodes | steps | GPU busy | ms/step | ps/node-step | render | receivers |
|---|---|---|---|---|---|---|---|
| RoomShoebox | 0.53M | 36,410 | 0.57s | 0.0157 | 29.6 | 0.6s | 5 |
| RoomChurch | 21.08M | 12,744 | 12.95s | 1.016 | 48.2 | 12.9s | 6 |
| RoomConcertHall | 57.28M | 9,103 | 25.30s | 2.779 | 48.5 | 25.2s | 16 |
| RoomShoeboxFcc | 0.13M | 15,416 | 0.118s | 0.0077 | 60.7 | 0.2s | 5 |
| RoomChurchFcc | 4.41M | 5,396 | 1.115s | 0.2066 | 46.9 | 1.2s | 6 |

Every scene is GPU-bound with no host gap: the church spends 12.95 s of GPU busy inside a 12.9 s render that also loads the scene, and the FCC church 1.115 s inside 1.2 s.
Encoding is not on the critical path either — a whole run's dispatches encode in 8 to 21 ms, against seconds of execution.

## What the step costs, and what it could

A step of this scheme is a stream, not a calculation.
The interior update reads the previous level once a node and reads and writes the level it is computing, which is twelve bytes a node-step, and everything else it does — six or twelve neighbour reads, the adjacency test, the shell — either hits cache or costs no memory at all.
So the question a room solver's performance reduces to is what fraction of the memory system a pass reaches, and `RoomTest --roofline` answers it directly: it times the interior kernel against `RoomStream`, which moves those same twelve bytes a node-step with none of the work, over the same buffers, back to back.

| scene | a pass moves | two-level stream | interior update | update / stream |
|---|---|---|---|---|
| RoomChurch | 247 MB | 401 GB/s | 370 GB/s | 92% |
| RoomConcertHall | 675 MB | 403 GB/s | 373 GB/s | 93% |
| RoomChurchFcc | 51 MB | 670 GB/s | 461 GB/s | 69% |

The frequency-dependent boundary gets a ceiling of its own from the same rung, because the grid pass's is not one it could ever reach: it gathers and scatters over a subset of the nodes across eight streams rather than two, and what it moves a lossy node is that node's own scalars, a read-modify-write of `u0` at a grid index, and the LRC velocity and integral of each of its material's branches — 176 of those ~201 bytes at the eleven branches every scene here uses.
`RoomLossyStream` moves exactly those, over the same nodes, in the same two-loop shape, with none of the arithmetic.

| scene | lossy nodes | a pass moves | boundary stream | boundary update | update / stream |
|---|---|---|---|---|---|
| RoomChurch | 496k of 949k | 100 MB | 286 GB/s | 287 GB/s | 100% |
| RoomConcertHall | 1.45M of 2.54M | 292 MB | 307 GB/s | 307 GB/s | 100% |
| RoomChurchFcc | 150k of 277k | 30 MB | 562 GB/s | 554 GB/s | 98% |

**The boundary pass is at its own roof on every scene that has one**, which is the answer to the obvious question about it: 287 GB/s next to a grid pass's 370 looks like a shortfall and is not one.
A scattered eight-stream pass simply does not reach a dense two-stream rate, and this one is within measurement of everything its access pattern allows.

**The ceiling is a property of the footprint, not of the machine.**
A stream over the church's 247 MB reaches 401 GB/s and over the hall's 675 MB reaches 403, while the FCC church's 51 MB reaches 670 — the smaller grid is partly served by cache, and a full-bandwidth room will sit at the 400 GB/s end whatever the scheme.
That is the number a step should be judged against, and it is roughly half of what the same machine gives a few-megabyte working set.

The two shoeboxes are not on that table because at 6 MB and 1 MB a pass is latency-bound rather than bandwidth-bound: the stream itself measures 706 and 120 GB/s, the second of which is below what the real step achieves, because 200 back-to-back dispatches over 0.13M nodes are measuring dispatch cost and not the memory system.
Their step cost is dispatch count, which is why they gained most when the step went from ten dispatches to two.

Counting the bytes a step cannot avoid — the two levels, the adjacency a node carries (a byte on the Cartesian grid, a quarter of one plus the packed mask at a boundary node on the FCC grid), and the impedance branches' own state at a lossy node — the church moves 372 MB a step and the hall 1031 MB, which against their measured times is **366 and 371 GB/s, or 91% of each grid's stream ceiling**.
The FCC church moves 86 MB a step at 437 GB/s, 64% of its ceiling.
Where the two grids' steps go is worth putting side by side, because they are limited by different things.
On the Cartesian church the two passes measured alone come to 0.667 and 0.348 ms against a 1.016 ms step: they account for it exactly, and both are at their own roofs, so that step is finished.
On the FCC church they come to 0.110 and 0.055 ms against a 0.197 ms step, and the 0.03 ms that is in neither is the drain between dependent dispatches — the interior pass writes the `u0` the boundary pass then reads.
That is why the FCC grid gained from going four dispatches to three and the Cartesian one did not: its dispatches are 6-17x shorter, so a fixed cost between them is a real share rather than noise.
Its interior pass at 69% of stream is the other half, and that is the rigid boundary's twelve masked neighbour reads at 6.3% of the nodes, which is arithmetic and divergence rather than traffic.

### How it scales

Every scene here is validation-scale, so the rung takes `--cells` as well as `--scene`: a grid of nothing but air, at any size, where every wall is the absorbing shell and there are no boundary nodes to build.
It answers the one question the scenes cannot, which is whether a node-step still costs what it costs once a grid is far larger than any of them.

| grid | nodes | state | two-level stream | interior update | update / stream | ps a node-step |
|---|---|---|---|---|---|---|
| 256³ | 16.8M | 0.13 GB | 397 GB/s | 386 GB/s | 97% | 31.1 |
| 512³ | 134M | 1.07 GB | 384 GB/s | 363 GB/s | 94% | 33.1 |
| 700³ | 343M | 2.74 GB | 363 GB/s | 345 GB/s | 95% | 34.8 |
| 900³ | 729M | 5.83 GB | 371 GB/s | 348 GB/s | 94% | 34.5 |
| 1000³ | 1.00G | 8.00 GB | 378 GB/s | 353 GB/s | 93% | 34.0 |
| 1280³ | 2.10G | 16.78 GB | 354 GB/s | 325 GB/s | 92% | 37.0 |

**A node-step costs what it costs.** The only real step is 256³ to 512³, which is a working set leaving cache for DRAM, and from there to 2.1 billion nodes the cost moves 12% while the grid grows 16×.
The update holds 92-97% of its own stream throughout.

The last row is the ceiling: at 2,147,483,647 nodes the 32-bit node indexing the kernels use runs out, and `RoomGpu::Init` throws above it.
That is 16.8 GB of state against this machine's 22.61 GB per-buffer limit and 30.15 GB recommended working set, so **the indexing binds before the memory does** on the explicit schemes.

An air grid also isolates what a boundary costs, and the answer reframes the tables above: 34 ps a node-step here against the FCC church's 46.9 is the church's 6.3% of boundary nodes and its lossy pass.
**Boundary nodes are a surface and interior nodes are a volume**, so that share collapses as a room is resolved finer — at 2.23 mm a 3.4 × 2.8 × 2.3 m room is 0.5% boundary nodes against the FCC church's 6.3%.
A full-bandwidth room is therefore closer to the air grid's cost than to any scene here.

`ACOUSTIC_ROOM_KERNEL_TIMES=1` puts each dispatch in its own command buffer and times it, which is how a step is split between kernels.
On the Cartesian church the serialized total comes to 1.024 ms/step against 1.016 uninstrumented, so the attribution is the real one; on the FCC church it comes to 0.585 against 0.197, because a command buffer's own fixed cost dominates a dispatch that short — those shares are directional only.

| kernel | RoomChurch | RoomChurchFcc |
|---|---|---|
| interior update, with the rigid boundary folded in | 68.0% | 69.1% |
| frequency-dependent boundary | 31.7% | 29.5% |
| receivers and source | 0.4% | 1.5% |

## Cartesian against FCC, at equal output bandwidth

The same room twice, at the same fmax, each scheme at its own points per wavelength: 10.5 Cartesian and 7.7 FCC, which is the pairing that puts both at **1.0% worst-direction phase-velocity error at fmax** — 1.0045% and 0.9929% from their own dispersion relations.
GPU busy from `ACOUSTIC_PROFILE=1`, taken from an interleaved run on a quiet M5 Max:

| | RoomShoebox | RoomShoeboxFcc | RoomChurch | RoomChurchFcc |
|---|---|---|---|---|
| stencil | 7-pt cart | 13-pt FCC | 7-pt cart | 13-pt FCC |
| fmax | 1 kHz | 1 kHz | 700 Hz | 700 Hz |
| grid spacing | 32.69 mm | 44.57 mm | 46.69 mm | 63.67 mm |
| Courant number | 0.999/√3 | 0.999 | 0.999/√3 | 0.999 |
| nodes | 0.529M | 0.126M | 21.08M | 4.41M |
| step rate | 18.20 kHz | 7.71 kHz | 12.74 kHz | 5.40 kHz |
| steps | 36,410 | 15,416 | 12,744 | 5,396 |
| node-steps | 19.3 G | 1.95 G | 268.6 G | 23.8 G |
| µs/step, GPU | 15.7 | 7.3 | 1016 | 197 |
| ps/node-step | 29.6 | 57.4 | 48.2 | 44.6 |
| two-level state | 4.2 MB | 1.0 MB | 169 MB | 35 MB |
| render, wall | 0.6s | 0.2s | 12.9s | 1.1s |
| worst receiver vs its own fp64 truth | 40.73 dB | 38.76 dB | 48.91 dB | 48.60 dB |

**FCC wins, by an order of magnitude in wall clock, and the whole margin is fewer node-steps rather than a cheaper node-step.**
On the church it is 12.2× less GPU time and 11.7× less wall clock.
The arithmetic behind it: at equal fmax the FCC spacing is 10.5/7.7 = 1.36× coarser, so once the fold is counted it holds 2(10.5/7.7)³ = 5.1× fewer nodes per unit volume — 4.8× measured, the difference being the fixed three-and-a-half-cell halo, which is a larger share of a smaller grid — and the Courant number and the spacing together drop the step rate by √3(10.5/7.7) = 2.36×.
That is 11.3× fewer node-steps, and the remaining 8% is the FCC step being cheaper per node-step than the Cartesian one.
Cost per node-step is 48.2 ps Cartesian against 44.6 ps FCC, near enough the same number though the FCC stencil reads twelve neighbours where the Cartesian one reads six.
That is the roofline saying what it always says here: both kernels stream two pressure levels and the extra neighbour reads are cache hits, so bytes moved and not arithmetic sets the cost.
The two are not made of the same parts, though, and the roofline table above says how: the FCC pass runs at 461 GB/s against the Cartesian one's 370, but against a ceiling that is 1.7× higher, so it reaches 69% of its own stream where the Cartesian pass reaches 92% of its.
The FCC grid gives its bandwidth advantage back at its boundaries, which are a larger share of a coarser room — 6.3% of its nodes are boundary nodes against 4.5%, and 3.4% lossy against 2.4% — and which carry twelve masked neighbour reads each.
Peak state falls by the same 4.8×, which matters more than the time does for the full-bandwidth rooms that do not fit at all on the Cartesian grid.

Two things the margin is not:

- **Not the small end.** The FCC shoebox gets 4.8× of GPU time and 3.0× of wall clock, because at 0.13M nodes a step is latency and dispatch count rather than memory traffic — 7.3 µs over an eighth of a million nodes against 15.7 µs for four times the nodes. The ratio approaches the arithmetic one as the room grows.
- **Not equal accuracy at the walls.** The pairing equalises *interior* dispersion, and nothing else. At equal fmax the FCC grid's spacing is 1.36× coarser in absolute terms, so a room's surfaces are staircased more coarsely, and the FCC rigid wall is not the exact half-cell mirror that the Cartesian one is on an axis-aligned box: the FCC shoebox's modes sit 0.9–2.0% low where the Cartesian shoebox's sit within 0.033%, all of it the wall. Through the WAV chain the two church discretisations still agree on the room — T20 1.88–2.24 s Cartesian against 1.51–2.35 s FCC over the six receivers — but the FCC one leaves 3.3% of its energy in the last tenth of the record against 0.2%.

So: FCC for bandwidth, and it is not close; the Cartesian grid keeps the sharper boundary at a given fmax, and on an axis-aligned box keeps an exactly analytic one.

## Implicit schemes

`RoomTest --implicit` measures the optimised implicit scheme of [Smits & Bilbao 2025](https://doi.org/10.1121/10.0036229) against the two explicit schemes the solver ships.
It is an instrument, not a stepper: nothing in `src/Room` runs the implicit scheme on a scene, and the reason is at the end of this section.

The scheme is 27-point and compact.
Both of its Laplacians are weighted sums of the same three sub-stencils — the 7-point face one, the 13-point edge one and the 9-point corner one — one weighting standing on the left of the time difference and the other on the right, so the update solves `(1 + Lim) u^(n+1) = (2 + 2 Lim + λ² Lex) u^n − (1 + Lim) u^(n−1)`.
The weights are free parameters, and the paper optimises them to hold phase error inside a threshold over as wide a band as possible rather than to raise the order of accuracy.
The `Lim` operator's diagonal dominance is what makes the solve cheap: the system is solved by a **fixed number of Jacobi sweeps**, not by ADI or tridiagonal factorisation, so there is no data dependency along a line anywhere and every pass is a plain stream.
A step is one pass to build the iteration's constant vector and seven sweeps — 3(P + 1) = 24 streams against an explicit step's 3 — over four grids of state against an explicit scheme's two.

Two things `--implicit` checks before it times anything.
On a periodic box a plane wave is an exact discrete eigenvector, so seeding both levels with one makes the next level read back 2 cos(ωT) − 1 at every node, and one step measures the scheme's dispersion relation exactly.
Measured that way, the realisation — seven Jacobi sweeps, single precision — sits within **5.0e-5 in phase velocity** of the relation its own float coefficients give, and 2000 steps at the worst-direction mode reach a peak of 1.802991 against the exact 1.802989 with a worst deviation of 2.4e-4, which is single-precision accumulation and not the truncated solve.
Sweeping the mode index out in frequency then gives the oversampling ratio the scheme needs, and for the 1%-threshold parameter set that is **ξ = 1.60, or 2.68 points per wavelength** — the paper's Table II value.

One correction the instrument makes to the paper: it recomputes the Courant limit from the scheme's own parameters (Eq. 16) rather than reading Table II's `λ_max` column, which is rounded to three places.
The 0.5% row's tabulated 0.850 is above its own limit of 0.8494, and at 0.850 a free-space mode grows by four orders of magnitude in 2000 steps.

Against the church, at the 1% worst-direction phase error all three schemes are pinned to:

| | Cartesian | FCC | implicit, 1% |
|---|---|---|---|
| points per wavelength | 10.5 | 7.7 | 2.68 |
| Courant number | 0.999/√3 | 0.999 | 0.840 |
| grid spacing | 46.7 mm | 63.3 mm | 183 mm |
| nodes | 21.08M | 4.41M | 0.361M |
| step rate | 12.74 kHz | 5.40 kHz | 2.23 kHz |
| steps for one second | 12,744 | 5,396 | 2,233 |
| node-steps | 268.6 G | 23.8 G | 0.81 G |
| streams a node-step | 3 | 3 | 24 |
| ps/node-step | 48.2 | 46.9 | 297 |
| GPU busy, one second of impulse response | 12.95s | 1.12s | 0.24s |
| state | 169 MB | 35 MB | 5.8 MB |

**The implicit scheme wins the interior, by 4.7× of GPU time and 6.1× of state against FCC.**
The implicit column is the interior alone — the other two include their boundary passes — and it is a projection from a measured step: `--implicit` times the church's own 21.0 × 13.7 × 7.4 m room at the scheme's 183 mm spacing, 116 × 76 × 41 nodes, at 0.1074 ms a step, and the FCC church is 1.12s of `gpu/exec` measured interleaved with it.
The paper's own benchmark on an RTX 4090 puts the same comparison at 4.0× with boundaries included on both sides, and its measured boundary overhead for this scheme is 27%, which would leave 3.7× here.

The whole margin is node-steps again, and more of it than FCC's was: 29× fewer than FCC, given back 6.3× by a step that costs 24 streams instead of 3.
In bytes the implicit step moves 77.5 GB against the FCC step's 459 GB, a ratio of 5.9× where the time ratio is 4.7×, and the difference is footprint: at 4.3 MB a pass the implicit church's passes reach 323 GB/s where the FCC church's reach 409.
The scheme does stream properly at scale — over a 384³ box, 680 MB a pass, a step runs at 334 GB/s, which is 81% of that footprint's 413 GB/s two-level stream, in the same band as the explicit church's 91%.
The one pass that falls short is the right-hand side, at 63% of the stream: it reads two grids with 27-point neighbourhoods rather than one, and two independent gather patterns do not cache as well as one.

### A rigid box, and what a boundary costs

`--implicit` also runs the scheme in a rigid box rather than a periodic one, which is the rung that would say whether it is a room solver.
The wall is the Neumann image the explicit schemes use for their halo: a step that would leave the grid comes back to the node one step inside, so a grid of N nodes spans (N−1)h and the discrete modes are cos(mπi/(N−1)).
That image makes the mode an exact discrete eigenvector the same way the periodic wrap makes a plane wave one — at i = 0 both neighbours are i = 1 and the cosine's evenness supplies the identity, at i = N−1 both are N−2 and cos(mπ − k) = cos(mπ)cos(k) supplies it again — so one step reads 2cos(ωT) − 1 back at the corner and the measurement is exact.

**The interior is right.** Over the 16 lowest modes of a 40 × 33 × 28 box the scheme lands within **0.026% of Morse & Ingard**, against the explicit Cartesian scheme's 0.033% and the FCC scheme's 0.9–2.0%, and within **1.2e-5** of the relation its own floats give, which is the Jacobi truncation and single precision together.
At 2.68 points per wavelength it resolves a room's modes as well as a 10.5-point grid does.

**The wall is unstable, and not for want of a smaller Courant number.**
Seeded with noise so every eigenvalue is excited, and judged on whether the envelope of max|u| in the last tenth of a run matches the first tenth:

| λ | of the free-space limit | envelope | |
|---|---|---|---|
| 0.8400 | 100% | 18.4 → 101.4 | grows |
| 0.7980 | 95% | 4.80 → 16.7 | grows |
| 0.7560 | 90% | 3.59 → 7.11 | grows |
| 0.7140 | 85% | 4.11 → 3.83 | holds |
| 0.6720 | 80% | 3.24 → 2.96 | holds |
| 0.5880 | 70% | 2.89 → 5.89 | grows |
| 0.5040 | 60% | 2.19 → 2.46 | grows |
| 0.4200 | 50% | 2.53 → 3.27 | grows |

**A Courant condition is monotone and this is not**, so lowering λ is not the fix — half the free-space limit still grows.
A single mode hides it for a long time: at the published λ the mode (0,1,1) reads 1.0026 against an exact 1.0020 after 5,000 steps and 1.038 after 20,000, then 10⁶ by 80,000, because what grows is an eigenvalue the seed barely projects onto until round-off finds it.
That is why the sweep seeds noise, and it is worth keeping in mind for any invented boundary: a mode soak that looks clean for twenty thousand steps is not evidence.

The reason is visible in the operator. The periodic stencil the paper's stability analysis covers is symmetric; the image is not.
At an edge node the step off the grid doubles a neighbour's weight, so the matrix carries a row of weight 2 against a column of weight 1 and is symmetric only in a weighted inner product — its spectrum is not the one Eq. 16's limit was derived for, and nothing about λ_max carries over.

**So a rigid room is not free either**, which is the practical finding.
The paper's own boundary is an admittance condition derived inside this lineage's energy framework, which is what makes such conditions provably stable, and it is not implemented here — substituting an image for it does not work at any Courant number tried.
Integrating this scheme therefore starts one step earlier than the frequency-dependent walls below: it starts at the published boundary condition, on the rigid case, where there is at least a formulation to follow.

**It is not integrated, and the obstacle is the walls rather than the arithmetic.**

- **The scheme's boundary condition is frequency-independent.** The paper's admittance γ is a real scalar, and its concluding remarks name frequency-dependent boundaries as future work. Every wall in every scene here is a set of LRC branches — eleven of them per material on the church — so the published scheme cannot render any scene in `Scenes/` but the two rigid shoeboxes. Combining the node-local branch solve with a Jacobi-iterated interior is unpublished work with no reference to check against.
- **There is no golden at that grid step.** Every gate in `script/ValidateRoom` is an SNR against PFFDTD's own output on the *same* voxelized input. At 183 mm the discrete problem is a different one, so there is no CUDA golden, no fp64 truth, and no bit-identity argument — the validation apparatus of the whole solver does not transfer.
- **2.68 points per wavelength is the geometry as well as the field.** The pairing above equalises interior dispersion and nothing else, the same caveat the Cartesian-against-FCC verdict carries, but three times as coarse. The paper compensates the absorption bias staircasing causes, and demonstrates it on a rotated rectangular room; nothing in it claims a complex room is the same room at 183 mm as at 47.

## Rendering

`script/RenderRoomWavs` turns a render into `gen/wav/room/<Scene>/R00n.wav` with the chain PFFDTD's own post-processing applies: the integrator folded into a 10 Hz fourth-order Butterworth low-cut (the analog high-pass with one of its four zeros at DC removed, then bilinear-transformed), a polyphase resample to 48 kHz, and Hamilton's DAFx 2021 approximate Stokes Green's-function air-absorption filter (see [LiteratureReview.md](LiteratureReview.md)) at the scene's own 20 °C and 50% relative humidity, truncated 120 dB down.
Receivers are normalised together so their relative levels survive, and written as float32 because a room impulse response covers more range than 16 bits.

What that leaves is one to two seconds of impulse response per scene, which is what the scenes were built for and is shorter than the rooms ring:

| scene | T20 | energy in the last tenth of the record |
|---|---|---|
| RoomShoebox | 2.16–2.38s | 8.7% |
| RoomChurch | 1.88–2.24s | 0.2% |
| RoomConcertHall | 1.78–2.15s | 1.8% |
| RoomShoeboxFcc | 2.95–3.67s | 4.3% |
| RoomChurchFcc | 1.51–2.35s | 3.3% |

T20 rather than T30 because the last 15 dB of a Schroeder curve over a record this short is mostly the truncation, and the tail share is the check on that.
The two shoeboxes are the case the tail share is there to catch: their walls are all rigid, so they do not decay at all and their apparent T20 is the record ending — which is why the two schemes disagree about it and neither number means anything.
The church and the hall are within a quarter second of each other, the church slightly the longer of the two, and the FCC church's six receivers span the Cartesian church's range.
