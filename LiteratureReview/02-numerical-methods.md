# Numerical Methods Literature (FDTD, PML, moving boundaries, GPU, validation)

Researched 2026-08-15. Unverified details are flagged inline and collected at the end.

## Staggered-grid FDTD for acoustics

- **Yee (1966)**, "Numerical Solution of Initial Boundary Value Problems Involving Maxwell's Equations in Isotropic Media," *IEEE Trans. Antennas Propag.* 14(3), 302–307. PDF: https://home.cc.umanitoba.ca/~lovetrij/cECE7810/Papers/Yee%201966%20HiRes.pdf
  The origin of the staggered leapfrog grid. WaveBlender's layout (pressure at cell centers, velocities on faces, half-step offsets) is exactly the Yee lattice; the staggering is why first-order differences yield second-order accuracy — a property the port must preserve through any memory-layout changes.
- **Botteldooren (1994)**, "Acoustical finite-difference time-domain simulation in a quasi-Cartesian grid," *JASA* 95(5), 2313–2319. https://pubs.aip.org/asa/jasa/article-abstract/95/5/2313/622635
  First prominent transfer of Yee's scheme to linear acoustics; establishes the exact pressure/velocity update equations in the WaveBlender kernels.
- **Botteldooren (1995)**, "Finite-difference time-domain simulation of low-frequency room acoustic problems," *JASA* 98(6), 3302–3308. https://pubs.aip.org/asa/jasa/article-abstract/98/6/3302/697760
  Founding paper of FDTD room acoustics: sources, impedance boundaries, error vs. frequency (informs the points-per-wavelength margin behind WaveBlender's 88.2–120 kHz rates).
- **Taflove & Hagness (2005)**, *Computational Electrodynamics: The FDTD Method*, 3rd ed., Artech House. https://us.artechhouse.com/Computational-Electrodynamics-The-Finite-Difference-Time-Domain-Method-Third-Edition-P2342.aspx
  The standard FDTD desk reference: von Neumann stability, 3D Courant limit (λ = cΔt/Δx ≤ 1/√3), and the Gedney PML chapter with polynomial conductivity grading and σ_max selection. Best single reference for verifying the ported update coefficients.
- **Bilbao (2009)**, *Numerical Sound Synthesis*, Wiley. https://www.wiley.com/en-us/Numerical+Sound+Synthesis:+Finite+Difference+Schemes+and+Simulation+in+Musical+Acoustics-p-9780470510469 (companion: https://www2.ph.ed.ac.uk/~sbilbao/nss.html)
  FDTD for audio: dispersion and energy-based stability analysis at audio rates. The energy-method perspective is valuable for reasoning about fp32 drift and long-run stability on Metal.
- **Hamilton & Bilbao (2017)**, "FDTD Methods for 3-D Room Acoustics Simulation With High-Order Accuracy in Space and Time," *IEEE/ACM TASLP* 25(11). https://dl.acm.org/doi/abs/10.1109/TASLP.2017.2744799
  Definitive modern dispersion/stability analysis for the 7-point scheme WaveBlender uses, with quantified usable-bandwidth limits. Its validation experiments double as our validation recipe (below).
- **Kowalczyk & van Walstijn (2011)**, "Room Acoustics Simulation Using 3-D Compact Explicit FDTD Schemes," *IEEE TASL* 19(1), 34–46. PDF: https://pureadmin.qub.ac.uk/ws/files/12979832/double.pdf
  Systematic scheme comparison with stability/dispersion analysis and boundary formulations — cross-check for Courant numbers and boundary updates.
- **Courant, Friedrichs, Lewy (1928)** — original CFL paper. English translation: https://web.stanford.edu/class/cme324/classics/courant-friedrichs-lewy.pdf (completeness; the practical bound is in Taflove & Hagness / Bilbao).

## PML absorbing boundaries

WaveBlender implements a Berenger-style split-field PML (p = px + py + pz, per-axis attenuation, 8-cell layer, separate pressure/velocity weight tables).

- **Bérenger (1994)**, "A Perfectly Matched Layer for the Absorption of Electromagnetic Waves," *JCP* 114(2), 185–200. PDF: https://web.stanford.edu/class/ee256/Berenger1994.pdf
  The original split-field PML; the matching condition and theoretical reflection coefficient R(θ) are the basis for validating the 8-cell layer.
- **Yuan, Borup, Wiskin, Berggren, Eidens, Johnson (1997)**, "Formulation and Validation of Berenger's PML … for FDTD Simulation of Acoustic Scattering," *IEEE UFFC* 44(4), 816–822. (Flag: assembled from search summaries; double-check page range.)
  First explicit port of split-field PML to the first-order pressure/velocity acoustic system on a staggered grid — the exact PML family WaveBlender implements, with measured numerical reflection coefficients to compare against.
- **Qi & Geers (1998)**, "Evaluation of the Perfectly Matched Layer for Computational Acoustics," *JCP* 139(1), 166–183. https://www.sciencedirect.com/science/article/abs/pii/S002199919795868X
  Primary reference for expected reflection levels from an 8-cell PML vs. thickness/profile/angle.
- **Liu & Tao (1997)**, "The perfectly matched layer for acoustic waves in absorptive media," *JASA* 102(4), 2072–2082. https://asa.scitation.org/doi/10.1121/1.419657
  Independent statement of the acoustic staggered-grid PML update stencils, including split-pressure bookkeeping.
- **Collino & Monk (1998)**, "Optimizing the perfectly matched layer," *CMAME* 164(1–2), 157–171. https://www.sciencedirect.com/science/article/abs/pii/S0045782598000528
  Why the *discrete* PML reflects when the continuous theory says it shouldn't; optimal layer count and grading — applicable to tuning/explaining residual reflections after the port.
- **Oskooi, Zhang, Avniel, Johnson (2008)**, "The failure of perfectly matched layers…," *Opt. Express* 16(15). PDF: https://math.mit.edu/~stevenj/papers/OskooiZh08.pdf
  Clean framework for *measuring* PML reflection numerically — defines the curve a PML regression test should reproduce.
- **Gedney (1996)** uniaxial PML + Taflove & Hagness ch. 7 — the standard source for polynomial grading practice σ(x) = σ_max(x/d)^m, m ≈ 3–4. (Note: WaveBlender's current profile is a simple quadratic ramp with a `TODO: switch to better PML (e.g., Convolutional-PML)` comment in `GPUSolver.cu`.)

## Immersed / moving boundaries, ghost and fresh cells

- **Mittal & Iaccarino (2005)**, "Immersed Boundary Methods," *Annu. Rev. Fluid Mech.* 37, 239–261. https://www.annualreviews.org/content/journals/10.1146/annurev.fluid.37.061903.175743
  The standard survey/taxonomy. Positions WaveBlender's beta blending field as a diffuse/volume-fraction immersed boundary vs. sharp ghost-cell schemes. Follow-up survey (2023): https://www.annualreviews.org/content/journals/10.1146/annurev-fluid-120720-022129
- **Mittal et al. (2008)**, "A versatile sharp interface immersed boundary method for incompressible flows with complex boundaries," *JCP* 227(10), 4825–4852. Open-access: https://pubmed.ncbi.nlm.nih.gov/20216919/
  **The canonical ghost-cell + fresh-cell paper** — describes the fresh-cell treatment WaveBlender's extrapolation echoes (a fresh cell was a ghost cell last step; reuse its interpolation nodes to avoid pressure spikes). Best single reference for justifying and testing `_freshCellPressure/_freshCellVelocity`.
- **Udaykumar, Mittal, Rampunggoon, Khanna (2001)**, "A Sharp Interface Cartesian Grid Method for Simulating Flows with Complex Moving Boundaries," *JCP* 174(1), 345–380. PDF: https://engineering.jhu.edu/fsag/wp-content/uploads/2014/06/jcp-vol174.pdf
  CFD ancestry of the moving-boundary/phase-change cell problem.
- **Bilbao (2022)**, "Immersed boundary methods in wave-based virtual acoustics," *JASA* 151(3), 1627. https://pubs.aip.org/asa/jasa/article/151/3/1627/2838190
  The main acoustics-native immersed-boundary FDTD reference; companion 3D impedance paper (2023): https://pubs.aip.org/asa/jasa/article/154/2/874/2906526
- **Tolan & Schneider (2003)**, "Locally conformal method for acoustic FDTD modeling of rigid surfaces," *JASA* 114(5). https://pubs.aip.org/asa/jasa/article-abstract/114/5/2575/547692
  Accuracy comparison point: conformal cell-area weighting vs. rasterized-beta boundaries.

## GPU FDTD acoustics implementations

- **Savioja (2010)**, "Real-time 3D FDTD simulation of low- and mid-frequency room acoustics," *DAFx-10*. PDF: https://www.dafx.de/paper-archive/2010/DAFx10/Savioja_DAFx10_P43.pdf
  Foundational GPU-FDTD paper (NVIDIA Research): kernel structure, memory-bandwidth-bound behavior — the performance regime the Metal port will live in.
- **Webb & Bilbao (2011)**, "Computing room acoustics with CUDA," *ICASSP 2011*. https://www.semanticscholar.org/paper/b2a9ebc198386655f832b22de3e7f39aa215ca9a
  CUDA thread-blocking strategies for 3D FDTD (2D extended tile + shared memory fastest) — the most transferable reference for retuning WaveBlender's 8×8×8 blocks into Metal threadgroups if optimization is needed. Companion AES 130 paper (flag: author list unverified): https://www.researchgate.net/publication/277835176
- **ParallelFDTD** (Saarelma & Savioja 2014, Forum Acusticum) — repo verified: https://github.com/juuli/ParallelFDTD — MIT CUDA room-acoustics solver with voxelizer; behavioral reference implementation.
- **PFFDTD** (Brian Hamilton) — repo verified: https://github.com/bsxfun/pffdtd — MIT, C/CUDA, multi-GPU, 7-point + FCC schemes, impedance boundaries, **energy conservation checked to machine precision** — that energy-audit instrumentation is a directly reusable validation idea (catches indexing and precision bugs that waveform comparisons miss).
- **Mehra, Raghuvanshi, Savioja, Lin, Manocha (2012)**, "An efficient GPU-based time domain solver for the acoustic wave equation," *Applied Acoustics* 73(2). http://gamma.cs.unc.edu/GPUSOUND/ — not FDTD (adaptive rectangular decomposition) but a standard GPU time-domain accuracy/performance comparison point.
- Survey: "Survey of GPU Acceleration in Acoustics and Audio Signal Processing," *ACM Computing Surveys*, https://doi.org/10.1145/3830900 (flag: authors/year uncaptured).
- **No notable Metal or WebGPU FDTD acoustics codebase was found** — the field is CUDA-dominated. The nearest Metal precedents are outside room/animation acoustics: Gebraad & Fichtner's seismic FDTD (see 03-metal-porting.md) and BabelViscoFDTD (ultrasound). This port breaks new ground.

## Validation methodology and analytic solutions

- **Pierce (2019)**, *Acoustics*, 3rd ed., Springer. https://link.springer.com/book/10.1007/978-3-030-11214-1
  Closed forms for the pulsating-sphere/monopole benchmark at finite ka (ch. 4) — the right test for a solver driven by rasterized vibrating geometry.
- **Morse & Ingard (1968)**, *Theoretical Acoustics*, Princeton. https://press.princeton.edu/books/paperback/9780691024011/theoretical-acoustics
  Free-space Green's function (G = δ(t − r/c)/4πr) for point-source impulse tests; rigid-box eigenmodes f = (c/2)·√((l/Lx)² + (m/Ly)² + (n/Lz)²) for standing-mode tests.
- **Schneider, Wagner, Broschat (1998)**, "Implementation of transparent sources embedded in acoustic FDTD grids," *JASA* 103(1). https://pubs.aip.org/asa/jasa/article-abstract/103/1/136/561253
  Read before writing the point-source test: a hard/soft source in a discrete grid does not reproduce the continuous Green's function unless injected "transparently" — otherwise observed error gets misattributed to the solver.
- **Hamilton & Bilbao (2017)** validation experiments (shoebox modal frequencies vs. the analytic formula; free-space band-limited pulse vs. exact solution) are a ready-made protocol. Their dispersion curves predict the error the solver *should* show — matching CUDA and Metal error against the same analytic curve is a strong equivalence test. Practical rigid-box walkthrough (blog-grade): https://computational-acoustics.gitlab.io/website/posts/5-acoustic-modes-of-a-rectangular-room/
- **BRAS benchmark** — Brinkmann, Aspöck, Ackermann, Weinzierl, Vorländer (2021), "A benchmark for room acoustical simulation," *Applied Acoustics* 176. Paper: https://www.sciencedirect.com/science/article/abs/pii/S0003682X20309725 — Database: https://depositonce.tu-berlin.de/items/38410727-febb-4769-8002-9c710ba393c4 (Zenodo mirror: https://zenodo.org/records/8419919)
  Seven measured reference scenes isolating single phenomena. Overkill for kernel-port verification; right target for end-to-end accuracy claims.
- **Thydal, Pind, Jeong, Engsig-Karup (2021)**, "Experimental validation and uncertainty quantification in wave-based computational room acoustics," *Applied Acoustics* 178. https://www.sciencedirect.com/science/article/abs/pii/S0003682X21000323
  Template for structuring error metrics and comparison protocol (their solver is DG-FEM).

## Unverified details (carry-through from research)

- Yuan et al. (1997) page range; Liu & Tao end page; Wang et al. (2018) article number (109); AES 130 companion author list; ACM CSUR survey authors/year; Bilbao 2023 author list; WaveCloud (Sheaffer & Fazenda 2014) has no recoverable live code repository.
