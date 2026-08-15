# Primary Papers: WaveBlender and Its Lineage

Researched 2026-08-15. All URLs verified 200 unless flagged **[bot-blocked]** (correct link, publisher 403s automated fetchers) — no known-dead links.

## The WaveBlender paper (the system being ported)

**Xue, Wang, Langlois, James. "WaveBlender: Practical Sound-Source Animation in Blended Domains." SIGGRAPH Asia 2024 Conference Papers. doi:10.1145/3680528.3687696** (four authors exactly — Aronson appears only on the 2023 bubbles paper).

- Project page: https://graphics.stanford.edu/papers/waveblender/
- PDF (3.4 MB): https://graphics.stanford.edu/papers/waveblender/assets/waveblender.pdf — full-res (53 MB): …/waveblender_full.pdf
- Video: https://graphics.stanford.edu/papers/waveblender/assets/waveblender.mp4 · YouTube: https://www.youtube.com/watch?v=wFxERIZE5WQ (the supplement *is* the video — no separate archive)
- ACM DL: https://dl.acm.org/doi/10.1145/3680528.3687696 [bot-blocked]
- Dataset: https://graphics.stanford.edu/papers/waveblender/dataset/ (see 04-data-and-validation.md)

Core method notes (from the PDF — these are the equations the port must reproduce):
- Blended velocity update (Eq. 8b): `v^{n+1/2} = (1−β)(v^{n−1/2} − ∇p·Δt/ρ0) + β·v_b^{n+1/2}`, β(t) = smoothstep 3t²−2t³ over each blend window.
- Blended cells act as a "β-medium": density ρ0/(1−β), sound speed c²_β = (1−β)c0² — **CFL is automatically satisfied if the base grid's CFL is**, and impedance stays uniform in blended cells.
- Batched pipeline: rasterize at t1/t2, precompute shader samples at 44.1–48 kHz for all boundary points, linearly interpolate to the FDTD rate (88.2 kHz+) during timestepping.
- Split-field PML (Liu & Tao 1997), 8 cells, **fused into the main kernels** — deviating from the separate interior/PML kernels of Mehra 2012 / KleinPAT, which the authors found slower on modern hardware. Worth re-measuring on Apple GPUs.
- Reported up to ~1000× faster than the 2018 CPU wavesolver (RTX 4090; e.g. "2016 Pouring Faucet": 55 h → 2.53 min).
- Table 1 lists per-scene grid dims, cell sizes (1–14.3 mm), FDTD rates (44.1–615 kHz), blend rates (50–2000 Hz) — reference performance/config targets for the Metal port.

## Code repositories

- **WaveBlender** — https://github.com/kangruix/WaveBlender — MIT, C++17/CUDA/Eigen 3.4/libigl. Last pushed 2025-09-08, active. README's "9/8 Update" promises "a more comprehensive and useable sound rendering system later" — **not released as of 2026-08-15** (checked author's repos, Stanford pages, searches). The reference codebase is stable.
- **FluidSound** — https://github.com/kangruix/FluidSound — MIT, C++11, Eigen-only coupled-bubble oscillator solver (implementation of Xue et al. 2023, building on Langlois 2016). **Archived (read-only) 2026-05-07**, last push 2024-12-19 — vendor/fork rather than track upstream. Note: the local WaveBlender checkout has not initialized this submodule (`git submodule update --init --recursive`).

## Direct lineage (what each shader/mechanism descends from)

### The predecessor wavesolver and the "acoustic shader" abstraction
**Wang, Qu, Langlois, James. "Toward Wave-based Sound Synthesis for Computer Animation." SIGGRAPH 2018. doi:10.1145/3197517.3201318** — https://graphics.stanford.edu/projects/wavesolver/ (PDF, video, slides, own dataset).
The CPU system WaveBlender replaces, and the paper that **introduced the acoustic-shader abstraction** WaveBlender adopts verbatim. WaveBlender's rigid/shell/speaker shader formulations (modal-to-boundary transfer, trapezoidal acceleration integration) follow this paper; its benchmarks are head-to-head against it. Differences: velocities (not accelerations) prescribed, shaders on GPU, fully explicit (no sparse pressure solve), β-blending instead of per-step re-rasterization (fixes the 2018 "popping" on topology change). **Fresh-cell pressure** also follows this paper: Neumann condition `p_f − p_i = −ρ0·a_n·(x_f − x_i)` (WaveBlender Eq. 13).

### β-blending's direct ancestor
**Allen & Raghuvanshi. "Aerophones in Flatland: Interactive Wave Simulation of Wind Instruments." SIGGRAPH 2015.** — https://www.microsoft.com/en-us/research/publication/aerophones-flatland-interactive-wave-simulation-wind-instruments/
**The most important methodological ancestor**: introduced the per-cell time-varying β parameter (tone holes, 2D 128 kHz GPU solver) and time-varying PML. WaveBlender §4 derives its scheme as a fix to Aerophones' velocity weight `w_β = βΔt/((1−β)+βΔt)` (too fast near blend end, distorts moving sources) → WaveBlender sets `w_β = β` directly in the discrete form. WaveBlender also borrows Aerophones' air-damping model (σ = 0.002–0.005/Δt in the candy scenes) and rejected its non-split PML in favor of Liu & Tao split-field.

### Modal / rigid-body sound
- **O'Brien, Cook, Essl 2001** — http://graphics.berkeley.edu/papers/Obrien-SSF-2001-08/ — and **O'Brien, Shen, Gatchalian 2002** — http://graphics.berkeley.edu/papers/Obrien-SSR-2002-07/ — origin of physics-based sound in graphics; the 2002 paper is the root of the modal-oscillators-driven-by-contact-impulses pipeline WaveBlender's Modal shader implements (`q̈ + C̃q̇ + K̃q = Uᵀf(t)`).
- **James, Barbič, Pai. "Precomputed Acoustic Transfer." SIGGRAPH 2006.** — http://graphics.cs.cmu.edu/projects/pat/ — the precomputed source-to-ear family WaveBlender positions itself against (PAT needs free-space/static scenes; the live FDTD domain removes that).
- **Wang & James. "KleinPAT." SIGGRAPH 2019.** — https://graphics.stanford.edu/projects/kleinpat/ — GPU time-domain transfer precomputation; cited for GPU kernel-partitioning practice WaveBlender deviates from (see fused-PML note above).
- Acceleration noise (Point shader): **Chadwick, Zheng, James 2012a** (SCA, soundbanks) — http://www.cs.cornell.edu/projects/Sound/proxy/ — and **2012b** (SIGGRAPH, precomputed acceleration noise) — https://history.siggraph.org/learning/precomputed-acceleration-noise-for-improved-rigid-body-sound-by-chadwick-zheng-and-james/ — WaveBlender's point sources use their Hertz-contact profile `a(t) = (π/2τ)Δv·sin(π(t−t0)/τ)`.

### Bubble-based water
- **Zheng & James. "Harmonic Fluids." SIGGRAPH 2009.** — https://www.cs.cornell.edu/projects/HarmonicFluids/ — bubbles-as-monopole-sources paradigm; WaveBlender replaces its per-frame Helmholtz solves with direct FDTD propagation.
- **Langlois, Zheng, James 2016. "Toward Animating Water with Complex Acoustic Bubbles." SIGGRAPH 2016. doi:10.1145/2897824.2925904** — https://www.cs.cornell.edu/projects/Sound/bubbles/ — source of the `trackedBubInfo.txt` bubble-tracking format and the "2016 Pouring Faucet" scenes.
- **Xue, Aronson, Wang, Langlois, James. "Improved Water Sound Synthesis using Coupled Bubbles." SIGGRAPH 2023.** — https://graphics.stanford.edu/papers/coupledbubbles/ (PDF: assets/coupledbubbles.pdf; code = FluidSound) — coupled (not independent) bubble oscillators + a sample-and-hold GPU wavesolver whose limitations (re-simulation overhead, no smooth animation) motivated β-blending. WaveBlender's analytic monopole bubble model (Eq. 11) comes from here; bubble-to-boundary transfer sampled at 1 ms.

### Thin shells
- **Chadwick, An, James. "Harmonic Shells." SIGGRAPH Asia 2009.** — https://www.cs.cornell.edu/projects/HarmonicShells/ — **the model the ShellShader adopts directly** (WaveBlender §5.1.3), via the 2018 pipeline: precomputed per-animation vertex displacements/accelerations, averaged per boundary face, trapezoidally integrated to velocities (Cymbal, Metal Sheet Shake scenes). Also introduced FFAT maps (later optimized by KleinPAT).
- **Cirio, Qu, Drettakis, Grinspun, Zheng. "Multi-scale Simulation of Nonlinear Thin-Shell Sound with Wave Turbulence." SIGGRAPH 2018.** — http://www.cs.columbia.edu/cg/waveturb/ — cited alongside Harmonic Shells in the source lineage.
- **Chaigne & Lambourg 2001**, *JASA* 109(4) — cited in WaveBlender's limitations: one-way coupling misses air-loading effects on plate decay times. (Flag: not independently fetched.)

## Concept-specific implementation citations (from the paper's reference list)

- **Fresh cells**: named after the CFD literature — Mittal & Iaccarino 2005 survey (doi:10.1146/annurev.fluid.37.061903.175743 [bot-blocked]); **Cheny & Botella 2010** (LS-STAG, doi:10.1016/j.jcp.2009.10.007) is the closest-boundary-velocity approach WaveBlender *rejects* ("spurious divergence sources") — instead it solves a **global least-squares/QR minimization of fresh-cell divergence on the CPU** (10–1000 fresh cells per batch; a notable per-batch overhead in the paper's Fig. 7, relevant to Metal-port scheduling).
- **Rasterization**: Akenine-Möller 2005 triangle-box overlap (the vendored `tribox.h`). CPU-side today and dominant per-batch cost in fast-moving scenes; the paper's future work explicitly suggests moving it to GPU.
- **GPU stencil playbook**: Micikevicius 2009, "3D Finite Difference Computation on GPUs Using CUDA" (doi:10.1145/1513895.1513905) — the optimization reference the implementation follows (object-free PML region, split-pressure computed only for warps touching the PML). Its warp-level assumptions need simdgroup-level rethinking on Apple GPUs.
- **Dynamic grids** (alternative approach, cited): Willemsen et al. 2021 DAFx (doi:10.23919/DAFx51585.2021.9768286 [bot-blocked]) and 2022 JAES 70(9) (paywalled, transcribed from the reference list); Bilbao 2022 immersed-boundary JASA (see 02-numerical-methods.md).
- FDTD/grid texts cited: Inan & Marshall 2011 (*Numerical Electromagnetics*), Bridson 2008 (*Fluid Simulation* — MAC grid convention).

## Ecosystem status (2026-08-15)

- Kangrui Xue: Stanford CS PhD student — https://profiles.stanford.edu/kangrui-xue · https://github.com/kangruix · Scholar: https://scholar.google.com/citations?user=RSsXZokAAAAJ (note kangruix.github.io is 404 — don't cite).
- The promised next-gen system from the authors has not appeared; FluidSound is frozen by archival. Reference codebase is a finished research artifact — stable porting target.
- Adjacent recent work (different group, Peking University): **Jin, Zhu, Wang, Li. "SonicRadiation: A Hybrid Numerical Solution for Sound Radiation without Ghost Cells." arXiv:2508.08775 (Aug 2025)** — https://arxiv.org/abs/2508.08775 — attacks the same moving-boundary radiation problem; worth tracking.
