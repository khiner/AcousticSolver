# Data and Validation Plan

## The WaveBlender dataset (primary validation data)

https://graphics.stanford.edu/papers/waveblender/dataset/ (verified 2026-08-15). **All 10 scenes downloaded and extracted to `../WaveBlender/Scenes/` (1.5 GB, 2026-08-15); every scene has a `config.json` and the formats below were confirmed on disk.** Scenes by source type, with archive sizes:

| Type | Scenes | Input formats |
|---|---|---|
| Water (bubbles) | 2016 Pouring Faucet (164 MB), Glass Pour (75.4 MB), Paddle Splash (63.3 MB) | `trackedBubInfo.txt` bubble tracking, surface `.obj` meshes at 10 ms intervals |
| Rigid bodies (modal) | Blue LEGO Drop (148 MB), Spolling Bowl (135 MB) | `displace.txt`, `.tet` tet meshes, `.tet.obj` surfaces, `.geo.txt` vertex maps, `.modes` eigenmodes, `.impulses.txt` |
| Speaker (audio) | Cup Phone (0.72 MB, also in-repo), Talk Fan (1.91 MB), Trumpet (0.87 MB) | `*_anim.txt` + input `.wav` |
| Point force (accel. noise) | Candy Shake (2.71 MB), Candy Fill'er Up (15.1 MB) | `*ptsrcData.txt` impulses, `betas/*.txt` auxiliary β data |

The paper's Table 1 gives per-scene grid dims, cell sizes (1–14.3 mm), FDTD rates (44.1–615 kHz), and blend rates (50–2000 Hz) — the config matrix the port must handle, and the performance-comparison targets.

**Coverage gap**: the dataset contains no thin-shell scene (the paper's Cymbal and Metal Sheet Shake examples were not published), so the ShellShader has no reference input data — validate it with synthetic vertex-displacement animations, or accept code-review-level parity for that shader. There is also no scene exercising the Monopole or Density shader classes (trivial to synthesize).

Also available: the **Wang et al. 2018 wavesolver dataset** — https://graphics.stanford.edu/projects/wavesolver/dataset/dataset_table.html — same lab, predecessor formats; useful only if we extend beyond WaveBlender's scenes.

**FluidSound**: cloned locally at `../FluidSound` (the GitHub repo is archived, so vendor it into the Metal port). Note WaveBlender's build expects it at `../WaveBlender/Source/FluidSound` — symlink if reading/building the reference tree.

**Local PDF library** — `~/acoustic_solver_papers/`, named `YYYY_authors_short-title.pdf`:
- `2005_taflove-hagness_computational-electrodynamics-fdtd-3ed-book.pdf`
- `2009_bilbao_numerical-sound-synthesis-book.pdf`
- `2001_chaigne-lambourg_damped-impacted-plates-1-theory.pdf`
- `2022_bilbao_immersed-boundary-methods-virtual-acoustics.pdf` (title page confirms: solo author, JASA 151, 1627–1638 — resolves the flags in 01/02)
- `2022_willemsen-bilbao-ducceschi-serafin_dynamic-grid-fdtd.pdf`

## Validation strategy (three tiers)

### Tier 1 — Analytic unit tests (solver physics, no WaveBlender data needed)
Sources in 02-numerical-methods.md, §Validation:
1. **Free-space point source vs. Green's function** G = δ(t − r/c)/4πr — inject transparently per Schneider, Wagner, Broschat (1998) or observed error is the source model's, not the solver's. Error should track the Hamilton & Bilbao (2017) dispersion curves — matching CUDA and Metal error against the same analytic curve is a strong equivalence test.
2. **Rigid-box standing modes** vs. f = (c/2)·√((l/Lx)² + (m/Ly)² + (n/Lz)²) (Morse & Ingard).
3. **Pulsating sphere / monopole radiation** at finite ka (Pierce ch. 4) — exercises the rasterized-vibrating-geometry path end to end.
4. **PML reflection coefficient** — measure per Oskooi et al. (2008) methodology; expected levels for an 8-cell layer from Qi & Geers (1998).
5. **Energy audit** — PFFDTD (github.com/bsxfun/pffdtd) checks energy conservation to machine precision; the same instrument catches indexing/precision bugs that waveform comparisons miss.

### Tier 2 — CUDA-vs-Metal cross-validation (port correctness)
- **Bit-for-bit equality is unattainable** (nvcc FMA contraction vs. Metal's compiler; Metal fast-math is on by default — build validation configs with `MTLCompileOptions.mathMode = safe`). Use tolerance-based comparison: relative L2/L∞ on pressure fields and listener audio over N timesteps.
- Comparison channels already in the codebase: listener `.bin` output (float32 pressure series) and `logZSlice` pressure+β slice dumps — run identical scenes on both backends and diff. A CPU fp32 reference implementation with controlled FMA is the most defensible ground truth for triage (the repo's `CPUSolver.h` is declared but "NOT IMPLEMENTED" — building it would serve both as reference and as the CI fallback).
- Progression: per-kernel golden tests (one timestep, synthetic fields) → per-batch (includes rasterization, fresh cells, shaders) → full scenes → audio-level metrics (spectrogram distance, e.g. multi-resolution STFT error) on dataset scenes vs. CUDA-produced outputs.

### Tier 3 — End-to-end perceptual/benchmark validation
- Dataset scenes compared against the paper's supplemental video audio and CUDA-run outputs.
- If accuracy claims beyond "matches the CUDA reference" are ever needed: BRAS measured benchmark (https://depositonce.tu-berlin.de/items/38410727-febb-4769-8002-9c710ba393c4) and the Thydal et al. (2021) UQ methodology — overkill for the port itself.

## Tooling for validation

- `scripts/write_wav.py` (in-repo): `.bin` → WAV at the FDTD rate.
- `scripts/run_all.py`: batch scene runner — the natural harness to extend for A/B backend comparison.
- Metal-side numerics controls: see 03-metal-porting.md §Precision (mathMode, `-fno-fast-math`, `precise::` namespace).
