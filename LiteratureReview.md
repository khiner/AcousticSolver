# Literature and Resources

Reference material behind this solver: the papers each part of the method comes from, the numerical-methods
background, the Metal/Apple documentation used while building it, and where the scene data lives.
Links verified 2026-08-15 (Bilbao room-acoustics section 2026-08-23).

## The paper being implemented

**Xue, Wang, Langlois, James. "WaveBlender: Practical Sound-Source Animation in Blended Domains."
SIGGRAPH Asia 2024.** doi:10.1145/3680528.3687696

- Project page: https://graphics.stanford.edu/papers/waveblender/
- PDF: https://graphics.stanford.edu/papers/waveblender/assets/waveblender.pdf (full-res: `waveblender_full.pdf`)
- Video: https://graphics.stanford.edu/papers/waveblender/assets/waveblender.mp4 ·
  https://www.youtube.com/watch?v=wFxERIZE5WQ (the supplement *is* the video)
- ACM DL: https://dl.acm.org/doi/10.1145/3680528.3687696
- Reference implementation: https://github.com/kangruix/WaveBlender — MIT, C++17/CUDA/Eigen 3.4
- Coupled-bubble solver: https://github.com/kangruix/FluidSound — MIT, archived read-only 2026-05-07
  (vendored into `src/`, so archival is not a concern)

Equation pointers used repeatedly: blended velocity update Eq. 8b, β-medium properties §4, analytic monopole
bubble model Eq. 11, fresh-cell Neumann pressure Eq. 13, split-field PML fused into the main kernels §6.
Table 1 lists per-scene grid dims, cell sizes (1–14.3 mm), FDTD rates (44.1–615 kHz), and blend rates
(50–2000 Hz).

## Method lineage

Where each mechanism in the solver comes from.

- **Wang, Qu, Langlois, James. "Toward Wave-based Sound Synthesis for Computer Animation." SIGGRAPH 2018.**
  https://graphics.stanford.edu/projects/wavesolver/ — the CPU predecessor that introduced the acoustic-shader
  abstraction, the modal/shell/speaker boundary formulations, and the fresh-cell Neumann pressure condition.
- **Allen & Raghuvanshi. "Aerophones in Flatland." SIGGRAPH 2015.**
  https://www.microsoft.com/en-us/research/publication/aerophones-flatland-interactive-wave-simulation-wind-instruments/
  — origin of the per-cell time-varying β parameter and the air-damping model. WaveBlender §4 is a correction
  to its velocity weight `w_β = βΔt/((1−β)+βΔt)` → `w_β = β`.
- **O'Brien, Shen, Gatchalian 2002.** http://graphics.berkeley.edu/papers/Obrien-SSR-2002-07/ — root of the
  modal-oscillators-driven-by-contact-impulses pipeline behind the Modal shader
  (`q̈ + C̃q̇ + K̃q = Uᵀf(t)`). Predecessor: **O'Brien, Cook, Essl 2001**,
  http://graphics.berkeley.edu/papers/Obrien-SSF-2001-08/.
- **Chadwick, Zheng, James 2012.** http://www.cs.cornell.edu/projects/Sound/proxy/ — acceleration noise; the
  Point shader's Hertz-contact profile `a(t) = (π/2τ)Δv·sin(π(t−t0)/τ)`.
- **Zheng & James. "Harmonic Fluids." SIGGRAPH 2009.** https://www.cs.cornell.edu/projects/HarmonicFluids/ —
  bubbles as monopole sources.
- **Langlois, Zheng, James. SIGGRAPH 2016.** https://www.cs.cornell.edu/projects/Sound/bubbles/ — source of the
  `trackedBubInfo.txt` bubble-tracking format and the pouring scenes.
- **Xue, Aronson, Wang, Langlois, James. "Improved Water Sound Synthesis using Coupled Bubbles." SIGGRAPH 2023.**
  https://graphics.stanford.edu/papers/coupledbubbles/ — the coupled-oscillator model implemented by FluidSound.
- **Chadwick, An, James. "Harmonic Shells." SIGGRAPH Asia 2009.**
  https://www.cs.cornell.edu/projects/HarmonicShells/ — the model the Shell shader adopts. Adjacent:
  **Cirio et al. 2018**, http://www.cs.columbia.edu/cg/waveturb/.
- **James, Barbič, Pai. "Precomputed Acoustic Transfer." SIGGRAPH 2006.** http://graphics.cs.cmu.edu/projects/pat/
  and **Wang & James. "KleinPAT." SIGGRAPH 2019.** https://graphics.stanford.edu/projects/kleinpat/ — the
  precomputed-transfer family this approach is positioned against.
- **Akenine-Möller 2005** triangle–box overlap — the rasterizer's `tribox` test.
- **Micikevicius 2009, "3D Finite Difference Computation on GPUs Using CUDA."** doi:10.1145/1513895.1513905 —
  the GPU stencil playbook the CUDA reference follows.

**Jin, Zhu, Wang, Li. "SonicRadiation: A Hybrid Numerical Solution for Sound Radiation without Ghost
Cells." CAVW (Proc. CASA 2026)** — https://arxiv.org/abs/2508.08775. TDBEM (BDF2 convolution quadrature,
after Lubich 1994 and Banjai & Sauter 2008) for near-field boundary potentials coupled to scalar
second-order FDTD for far-field transport, with no ghost cells, no rasterization, and no fresh-cell
problem. Implemented in `src/Radiation/`, with its monopole validation ladder in the `RadiationTest`
target. No public reference implementation exists.

## The Bilbao room-acoustics lineage

**Goal: room impulse responses at full audio bandwidth** — a closed room on a regular grid, energy
leaving only through frequency-dependent impedance walls, with energy-based stability proofs. From
Stefan Bilbao's group (Edinburgh Acoustics and Audio Group: Brian Hamilton, Craig Webb, Jan Smits,
and collaborators), and implemented in `src/Room/` on the 7-point Cartesian stencil, with its
validation ladder in the `RoomTest` target. Unlike the two solvers above it has a
public reference implementation. Neither other solver covers this ground: WaveBlender animates
sources near objects and SonicRadiation radiates to the far field, and both dump energy into a PML —
here the walls themselves are the physics.

The core:

- **Bilbao, Hamilton, Botts, Savioja. "Finite volume time domain room acoustics simulation under
  general impedance boundary conditions." IEEE/ACM TASLP 24(1), 2016.** doi:10.1109/TASLP.2015.2500018 ·
  OA: https://www.pure.ed.ac.uk/ws/files/22154168/fv_genimp_final_r3.pdf — the canonical scheme:
  interior FDTD (7-point Cartesian or 13-point FCC) recast as finite volumes at the boundary, general
  LRC (frequency-dependent) wall impedance per boundary node, energy-stable. The algorithmic core of
  PFFDTD.
- **Hamilton. "Finite Difference and Finite Volume Methods for Wave-based Modelling of Room
  Acoustics." PhD thesis, U. Edinburgh, 2016.**
  https://www.brianhamilton.co/thesis/hamilton2016phdthesis.pdf — the single most complete exposition:
  grid/scheme families (Cartesian, FCC, CCP), dispersion, stability, and the FV boundary formulations.
- **Bilbao. "Modeling of complex geometries and boundary conditions in FD/FV time domain room
  acoustics simulation." TASLP 21(7), 2013.** doi:10.1109/TASL.2013.2256897 — origin of the
  locally-conforming FV boundary-cell idea.
- **Bilbao & Hamilton. "Passive volumetric time domain simulation for room acoustics applications."
  JASA 145(4), 2019.** doi:10.1121/1.5095876 — the passivity framing (lossless interior + passive
  wall-admittance feedback) the family's stability argument rests on.

Reference implementation:

- **PFFDTD** — https://github.com/bsxfun/pffdtd (Hamilton, MIT). Python voxelizer/setup + CPU
  reference engine + C/CUDA multi-GPU engine implementing the 2016 boundary scheme on 7-point
  Cartesian and 13-point FCC grids, with energy conservation checked to machine precision,
  single-precision stability safeguards, and a staircase surface-area correction. No journal paper —
  the software is the citation. Modernized fork (SYCL/C++, CMake, CI, actively maintained):
  https://github.com/tobanteAudio/pffdtd. PFFDTD plays the role the kangruix/WaveBlender CUDA repo
  played for the first solver — golden-output generator and behavioral reference — and its CPU engines
  run locally, so no remote GPU host is needed.

Modern upgrades:

- **Smits & Bilbao. "Optimised implicit finite-difference schemes for the wave equation with
  admittance boundary conditions." JASA 157(3), 2025.** doi:10.1121/10.0036229 · OA:
  https://www.pure.ed.ac.uk/ws/files/498268079/Bilbao2025JASAOptimised.pdf — the group's most recent
  solver paper: 27-point compact implicit schemes whose weights are optimised against a wideband
  dispersion criterion rather than for order of accuracy, reaching full 20 kHz bandwidth on grids
  far coarser than the explicit family allows. The implicit system is solved by a fixed number of
  Jacobi sweeps, not by ADI or factorisation, so every pass is a plain stream. Its boundaries are
  frequency-*in*dependent real admittances over staircased geometry, with the absorption bias
  staircasing causes compensated by effective surface areas; frequency-dependent boundaries are
  named as future work. Measured against both explicit schemes here under **Implicit schemes** in
  [VALIDATION.md](VALIDATION.md).
- **Smits. "Efficient FD room acoustics simulation incorporating extended-reacting elements." DAFx
  2023.** — porous absorbers via auxiliary ODEs.
- **Hamilton & Bilbao 2017** (high-order accuracy in space and time) — cited in the FDTD section
  below for its dispersion analysis and validation protocol.

Irregular geometry without staircasing — the immersed-boundary line, the treatment closest in spirit
to WaveBlender's blended domains: **Bilbao 2022** (cited below) plus the impedance-barrier pair,
**Bilbao 2023, "…acoustic barriers using the IBM: The one-dimensional case," JASA 153(4)**
doi:10.1121/10.0017763 and **"The three-dimensional case," JASA 154(2)** doi:10.1121/10.0020635 —
immersed surfaces with tunable impedance and transmittance. Applied downstream by **Wu, Mohapatra,
Fels, Interspeech 2024** (vocal-tract FDTD with immersed boundaries).

Air absorption:

- **Hamilton & Bilbao. "Time-domain modelling of wave-based room acoustics including viscothermal and
  relaxation effects in air." JASA-EL 1(9), 2021.** doi:10.1121/10.0006298 — losses inside the scheme.
- **Hamilton 2021**, arXiv:2107.11871 (modal air-attenuation for simulated RIRs) and DAFx 2021
  (approximate Stokes Green's-function filter) — the post-processing route PFFDTD takes instead.

Sources and receivers:

- **Bilbao & Hamilton. "Directional sources in wave-based acoustic simulation." TASLP 27(2), 2019.**
  doi:10.1109/TASLP.2018.2881336, and **Bilbao, Ahrens, Hamilton, JASA 146(4), 2019** — measured
  source directivity on the grid.
- **Bilbao, Politis, Hamilton. IEEE SPL 26(4), 2019** — local time-domain spherical-harmonic
  (ambisonic) receiver encoding in FDTD.

GPU implementation record (the CUDA playbook to translate to Metal):

- **Webb. "Parallel computation techniques for virtual acoustics and physical modelling synthesis."
  PhD thesis, U. Edinburgh, 2014** and **Webb & Bilbao, DAFx 2012** (binaural audio-rate FDTD in CUDA).
- **Hamilton & Webb, DAFx 2013** (FCC-grid FD/FV on GPU), **Hamilton, Bilbao, Webb, DAFx 2014**
  (implicit schemes on GPU), **Hamilton, Webb, Gray, Bilbao, DAFx 2015** (large stencils),
  **Hamilton, Webb, Fletcher, Bilbao, ISMRA 2016** (multi-GPU, viscothermal + impedance),
  **Stoltzfus, Gray, Dubach, Bilbao, DAFx 2017** (performance portability).

Survey: **Hamilton & Bilbao. "Wave-based room acoustics modelling: recent progress and future
outlooks." IOA Auditorium Acoustics, 2018.**

Not open-access anywhere: **Bilbao & Hamilton, JAES 65(1/2), 2017** (explicit/implicit FV with
viscothermal losses and frequency-dependent boundaries), doi:10.17743/jaes.2016.0057 — content
covered by the thesis, the 2019 passivity paper, and the 2021 JASA-EL letter.

## FDTD, PML, and immersed boundaries

- **Yee 1966**, *IEEE TAP* 14(3), 302–307.
  https://home.cc.umanitoba.ca/~lovetrij/cECE7810/Papers/Yee%201966%20HiRes.pdf — the staggered leapfrog grid.
- **Botteldooren 1994**, *JASA* 95(5) https://pubs.aip.org/asa/jasa/article-abstract/95/5/2313/622635 and
  **1995**, *JASA* 98(6) https://pubs.aip.org/asa/jasa/article-abstract/98/6/3302/697760 — Yee's scheme in
  linear acoustics; the exact pressure/velocity updates in the kernels.
- **Taflove & Hagness 2005**, *Computational Electrodynamics: The FDTD Method*, 3rd ed. — von Neumann stability,
  the 3D Courant limit `cΔt/Δx ≤ 1/√3`, and the PML chapter (polynomial grading `σ(x) = σ_max(x/d)^m`).
- **Bilbao 2009**, *Numerical Sound Synthesis* — dispersion and energy-based stability at audio rates.
  Companion: https://www2.ph.ed.ac.uk/~sbilbao/nss.html.
- **Hamilton & Bilbao 2017**, *IEEE/ACM TASLP* 25(11).
  https://dl.acm.org/doi/abs/10.1109/TASLP.2017.2744799 — dispersion/stability for the 7-point scheme used
  here, with usable-bandwidth limits and a ready-made validation protocol.
- **Kowalczyk & van Walstijn 2011**, *IEEE TASL* 19(1).
  https://pureadmin.qub.ac.uk/ws/files/12979832/double.pdf — scheme comparison, Courant numbers, boundary updates.
- **Bérenger 1994**, *JCP* 114(2). https://web.stanford.edu/class/ee256/Berenger1994.pdf — the original
  split-field PML.
- **Liu & Tao 1997**, *JASA* 102(4). https://asa.scitation.org/doi/10.1121/1.419657 — the acoustic staggered-grid
  split-field PML this solver implements.
- **Qi & Geers 1998**, *JCP* 139(1).
  https://www.sciencedirect.com/science/article/abs/pii/S002199919795868X — expected reflection levels vs.
  thickness/profile/angle for an 8-cell layer. **Collino & Monk 1998**,
  https://www.sciencedirect.com/science/article/abs/pii/S0045782598000528 — why the *discrete* PML reflects at all.
- **Oskooi, Zhang, Avniel, Johnson 2008**, *Opt. Express* 16(15).
  https://math.mit.edu/~stevenj/papers/OskooiZh08.pdf — how to measure PML reflection numerically.
- **Mittal et al. 2008**, *JCP* 227(10). https://pubmed.ncbi.nlm.nih.gov/20216919/ — the canonical ghost-cell +
  fresh-cell paper. Survey: **Mittal & Iaccarino 2005**,
  https://www.annualreviews.org/content/journals/10.1146/annurev.fluid.37.061903.175743 (2023 follow-up:
  `annurev-fluid-120720-022129`).
- **Bilbao 2022**, "Immersed boundary methods in wave-based virtual acoustics," *JASA* 151(3), 1627.
  https://pubs.aip.org/asa/jasa/article/151/3/1627/2838190 — the acoustics-native immersed-boundary reference.
- **Morse & Ingard 1968**, *Theoretical Acoustics* — free-space Green's function and rigid-box eigenmodes
  `f = (c/2)·√((l/Lx)² + (m/Ly)² + (n/Lz)²)`. **Pierce 2019**, *Acoustics* 3rd ed., ch. 4 — pulsating-sphere
  radiation at finite `ka`. **Schneider, Wagner, Broschat 1998**, *JASA* 103(1),
  https://pubs.aip.org/asa/jasa/article-abstract/103/1/136/561253 — transparent source injection, required
  before any point-source accuracy test means anything.
- Alternatives to a fixed grid, cited by the paper but not taken: **Willemsen, Bilbao, Ducceschi, Serafin**,
  dynamic-grid FDTD, DAFx 2021 doi:10.23919/DAFx51585.2021.9768286 and *JAES* 70(9), 2022.
- **Chaigne & Lambourg 2001**, *JASA* 109(4) — damped impacted plates; the air-loading effect on plate decay
  times that one-way coupling misses.

Other GPU FDTD acoustics implementations, for behavioral comparison (PFFDTD and the Webb/Hamilton
CUDA work are in the Bilbao section above):
**ParallelFDTD** https://github.com/juuli/ParallelFDTD (CUDA + voxelizer),
**Savioja 2010** https://www.dafx.de/paper-archive/2010/DAFx10/Savioja_DAFx10_P43.pdf.

## Metal and Apple silicon

- **Metal Shading Language Specification** — https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- **Metal Feature Set Tables** — https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
- **metal-cpp** — https://developer.apple.com/metal/cpp/ · https://github.com/apple/metal-cpp
- **Porting your Metal code to Apple silicon** —
  https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon
- **WWDC22 "Scale compute workloads across Apple GPUs"** — https://developer.apple.com/videos/play/wwdc2022/10159
  — GPU-timeline gaps and pipelining, the material behind this solver's batch encoding.
- **WWDC22 "Program Metal in C++ with metal-cpp"** — https://developer.apple.com/videos/play/wwdc2022/10160/
- **Tech Talk "Metal Compute on MacBook Pro"** — https://developer.apple.com/videos/play/tech-talks/10580/ ·
  **"Learn performance best practices for Metal shaders"** — https://developer.apple.com/videos/play/tech-talks/111373/
- **Creating threads and threadgroups** — https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups
- **Indirect command encoding** — https://developer.apple.com/documentation/metal/indirect-command-encoding
- **Capturing a Metal workload in Xcode** — https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-in-xcode
- Fast math is **on** by default in Metal (opposite of nvcc): `MTLCompileOptions.mathMode` —
  https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode, or `-fno-fast-math` to
  `xcrun metal`. There is no `double` in device code. Both facts shape the golden-output tolerances in
  [VALIDATION.md](VALIDATION.md).
- GPU timer units pitfall: https://feresignum.com/resolving-metal-gpu-timers/

Accelerate, for the CPU-side dense linear algebra in the bubble and modal paths:

- **BLAS/LAPACK** — https://developer.apple.com/documentation/accelerate/blas-library ·
  `ACCELERATE_NEW_LAPACK` headers: https://developer.apple.com/documentation/accelerate/using-the-latest-lapack-interfaces
- **corsix's AMX notes** — https://github.com/corsix/amx — the per-cluster sharing behavior that caps
  factorization throughput no matter the thread count.

Exemplar Metal codebases:

- **MLX** — https://github.com/ml-explore/mlx — idiomatic MSL + C++ host; `steel/` for `simdgroup_matrix` GEMM.
- **llama.cpp** `ggml/src/ggml-metal/` — https://github.com/ggml-org/llama.cpp — runtime `.metal` compilation,
  hundreds of dispatches per command buffer, no Xcode.
- **metal-benchmarks** — https://github.com/philipturner/metal-benchmarks — Apple GPU microarchitecture
  reference (latencies, caches, occupancy) for setting bandwidth expectations.
- **BabelViscoFDTD** — https://github.com/ProteusMRIgHIFU/BabelViscoFDTD — staggered-grid FDTD with CUDA,
  OpenCL, and Metal backends (ultrasound).
- **Gebraad & Fichtner 2023**, "Seamless GPU acceleration for C++-based physics with the Metal Shading Language
  on Apple's M-series chips," *SRL* 94(3). https://arxiv.org/abs/2206.01791 — the closest published precedent:
  a C++ elastic-wave FDTD solver in MSL on M1.

## Data

**WaveBlender dataset** — https://graphics.stanford.edu/papers/waveblender/dataset/ — 10 scenes, 1.5 GB
extracted, downloaded into `Scenes/` by `script/FetchScenes`, which also downloads two more
scenes (`Cymbal`, `WineglassTap`, ~10 GB) from the predecessor dataset below.

| Source type | Scenes | Input formats |
|---|---|---|
| Water (bubbles) | `2016Pour`, `GlassPour`, `PaddleSplash` | `trackedBubInfo.txt`, surface `.obj` at 10 ms intervals |
| Rigid bodies (modal) | `LegoDrop`, `SpollingBowl`, `WineglassTap` | `displace.txt`, `.tet`, `.tet.obj`, `.geo.txt`, `.modes`, `.impulses.txt` |
| Speaker (audio) | `CupPhone`, `TalkFan`, `Trumpet` | `*_anim.txt` + input `.wav` |
| Point force | `FillerUp`, `HandShake` | `*ptsrcData.txt`, `betas/*.txt` |
| Thin shell | `Cymbal` | per-frame `.displacement` and `.wsacc` (plus their constrained-vertex halves) at 44.1 kHz |

No published scene exercises the Monopole shader (a test source with no scene of its own).

The predecessor **Wang et al. 2018 dataset** —
https://graphics.stanford.edu/projects/wavesolver/dataset/dataset_table.html — same lab. Its thin-shell and
rigid-body files are in the formats our Shell and Modal shaders read. `Cymbal` (10 GB) and `WineglassTap`
(202 MB) come from here. The WaveBlender dataset page has both thin-shell rows commented out and Cymbal's
link blanked, so this is the only page that still links that data. Also available here, not used: Metal Sheet
Shake (40 GB, the other thin-shell example), Dripping Faucet and the 17 GB raw Pouring Faucet, and ABCD
(characters). The two water scenes would need work: their bubble data is per-timestep snapshots, and
converting it to the tracked `trackedBubInfo.txt` our Bubbles shader reads needs the Langlois et al. 2016
bubble tracker, which neither codebase includes. The paper's "2016 Water Step" was never published.

## Authors

Kangrui Xue — https://profiles.stanford.edu/kangrui-xue · https://github.com/kangruix ·
https://scholar.google.com/citations?user=RSsXZokAAAAJ
