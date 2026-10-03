# Acoustic Reliefs on Metal

This is an Apple Silicon implementation of [Acoustic Reliefs](https://doi.org/10.1145/3763287). The runner generates the mesh, optimizes the heightfield with the Metal acoustic solver and MPS appearance objective, and writes a final OBJ. Supply a target image, such as the Cat image from the authors' repository.

From the repository root:

```sh
cmake -S . -B build
cmake --build build --target BemTextureSolve BemSolve -j 6
python3 -m venv build/reliefs-venv
build/reliefs-venv/bin/pip install -r script/reliefs/requirements.txt
build/reliefs-venv/bin/python -m script.reliefs.run config/reliefs/cat.toml TARGET.png build/reliefs-cat
build/reliefs-venv/bin/python -m script.reliefs.run config/reliefs/cat.toml TARGET.png build/reliefs-cat-refined --refine build/reliefs-cat
```

Use `--steps N` for a bounded run. Rerun with a larger total or omit it to resume. Each run writes `hfield.npy`, `mesh.obj` when complete, checkpoints, and a short `summary.json`. The target image and run configuration must remain the same when resuming. The first run downloads the pinned CLIP weights.

The Metal solver uses compressed FP32 operators with FP64 host setup and sampled residual checks. `BemSolve --reference` provides a dense FP64 CPU check on small meshes. Run the generated-mesh sanity check with:

```sh
build/reliefs-venv/bin/python -m script.reliefs.sanity
```

## Upstream comparison

`compare_upstream.py` compares one saved heightfield on the same mesh and target image. Prepare a case on the Mac, copy its directory to a CUDA host that has this repository and the authors' pinned checkout, then capture the upstream CPU acoustic and CUDA appearance results:

```sh
build/reliefs-venv/bin/python -m script.reliefs.compare_upstream prepare build/reliefs-cat TARGET.png build/reliefs-compare
python -m script.reliefs.compare_upstream cuda build/reliefs-compare /path/to/acoustic_reliefs
```

Copy `upstream.json` back into the case directory and score it on the Mac:

```sh
build/reliefs-venv/bin/python -m script.reliefs.compare_upstream metal build/reliefs-compare
```

The script accepts the authors' released revision and the recorded ACA-corrected revision. It records which one ran. Acoustic time measures the authors' CPU BEM against this Metal BEM. Appearance time measures their CUDA renderer and CLIP against Mitsuba Metal and MPS. Compare errors on the matched state. Optimization trajectories may diverge with floating-point rounding.

Derived meshing and objective code retains the authors' [BSD license](../../src/Bem/LICENSE.AcousticReliefs). OpenAI CLIP retains its [MIT license](LICENSE.OpenAI-CLIP). The CLIP objective also credits [CLIPasso](https://github.com/yael-vinker/CLIPasso/blob/92262f702b6592b6c25c80def6284ad06225eadd/models/loss.py) under [CC BY-NC-SA 4.0](LICENSE.CLIPasso).
