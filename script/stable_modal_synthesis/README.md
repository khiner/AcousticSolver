# Stable Differentiable Modal Synthesis

Reproduction of Zheleznov, Bilbao, Wright, and King, [Stable Differentiable Modal Synthesis for Learning Nonlinear Dynamics](https://arxiv.org/abs/2601.10453v1), January 2026 preprint and [JAES 2026 article](https://doi.org/10.17743/jaes.2026.0269).
The [author implementation](https://github.com/victorzheleznov/jaes-modal-node) is pinned at `68017c953319636b7b69018d8e406f17e18cf0de`; [protocol.json](protocol.json) records source and checkpoint hashes, the released configuration, audio directories, and Table 2 values.

## Scope and status

This experiment learns nonlinear coupling from synthetic string modal displacement and velocity trajectories. It does not estimate an unknown scene from audio or video, infer contact topology, or jointly identify a contact/body model. Those remain separate research tasks; the DiffImpact reproduction remains in [../diffimpact](../diffimpact/README.md).

The target is the JAES model with exact spectral string nonlinearity, 75 modes, a 1,000-unit convex GradNet, and the scalar auxiliary variable (SAV) solver. The DAFx cubic string approximation is a different experiment.

Released-checkpoint evaluation and dataset regeneration are complete. Training stopped after 409 of the published 2,000 epochs and did not match the released checkpoint's quality; full local training has not been run. See [validation results](../../VALIDATION.md#stable-modal-synthesis) for numerical agreement and reproduction limits.

## Reproduction boundary

The CUDA path imports the unchanged author sources. It retains float64, the published physical distributions and split seeds, 60/20/60 train/validation/test recordings, 2 seconds at 88.2 kHz for training, 3 seconds at 96 kHz for validation/test, 1 ms teacher-forced training segments, and Adam at 0.001. The author configuration specifies 2,000 epochs; the current run ended after 409 completed epochs at the user's request. The loss is the released sum of displacement and velocity MSE. The SAV drift correction remains detached during backpropagation.

CUDA and CPU RNG streams generate different physical parameters from the same seed. The faithful CUDA reproduction generates and compares every normalized target and linear waveform against the release before training. The local pilot below instead regenerates targets from archived CUDA-drawn parameters, checks them against released audio, and records CPU arithmetic as an adaptation. Do not reuse a CPU-generated dataset just because its author directory hash matches: that hash does not encode the device.

Recorded execution adaptations:

- Python 3.12 instead of the author's documented 3.11.9, with pinned release package versions.
- One GPU with batch size four instead of four GPUs with batch size one. Effective batch size and updates per epoch are preserved; floating-point reduction ordering differs.
- The current run ended after 409 total epochs instead of 2,000, as requested by the user; completed epochs and optimizer state were retained across restarts.
- CUDA Graph capture of unchanged SAV steps for long inference. Training gradients use the original eager implementation.
- Per-epoch checkpoint/status retention and disabled external experiment logging.
- Separate process lifetimes for preparation, training, and final evaluation, preventing GPU allocations from overlapping across phases.
- Figure data, selected examples, and plotting functions follow the author notebook; typography and output formats differ.

`cuda_job.py` generates and fingerprints datasets, evaluates the released checkpoint, invokes the original trainer through `cuda_train.py`, then evaluates the newly trained checkpoint. Reports retain per-recording errors, means, ratios to the rounded paper values, audio, selected full states, and figures. Agreement must be assessed from those outputs; finishing a job alone does not certify reproduction.

## Local checks and native rendering

From the repository root:

```sh
git submodule update --init external/stable-modal-synthesis
PATH="$(brew --prefix llvm)/bin:$PATH" script/Build
python3 -m venv build/reference/stable-modal-env
build/reference/stable-modal-env/bin/pip install -r script/stable_modal_synthesis/requirements.txt
build/reference/stable-modal-env/bin/python -m script.stable_modal_synthesis.check \
  --output build/stable-modal-checks.json
build/reference/stable-modal-env/bin/python -m script.stable_modal_synthesis.render_check \
  --output build/stable-modal-render-full
```

Use a fresh output directory for each check or golden capture. `--seconds` explicitly selects a shortened diagnostic. Full durations are the default.

[Modal.metal](../../src/StableModal/Modal.metal) implements the spectral string generator; [Modal.cpp](../../src/StableModal/Modal.cpp) supplies the metal-cpp host. CMake builds the `StableModalTraining` and `StableModalMetal` libraries; the Python adapter loads them and prepares author weights and physical inputs. [FloatPair.h](../../src/Metal/FloatPair.h) shares float-pair arithmetic with the String and Coupled solvers: about 48 significand bits with float32 exponent range, not IEEE float64. Safe Metal math is required. Tests compare complete trajectories independently against the author's float64 implementation.

For a direct CUDA-to-native check, run `golden.py` on CUDA, copy its output directory to the Mac, then run:

```sh
build/reference/stable-modal-env/bin/python -m script.stable_modal_synthesis.render_check \
  --golden build/stable-modal-cuda-golden \
  --output build/stable-modal-render-cuda
```

The capture includes CUDA-computed mode shapes and excitation to isolate solver error from input preparation. The port currently fixes 75 modes and supports up to 1,000 hidden units.

## Native training

[TrainAccelerate.cpp](../../src/StableModal/TrainAccelerate.cpp) implements forward SAV execution, the discrete reverse pass, initial auxiliary-state derivative, GradNet parameter reductions, and Adam in float64. The drift correction remains detached as in the author implementation. The derivative of leaky-ReLU at zero uses the negative branch, matching PyTorch. Loss is the sum of mean displacement and velocity squared errors, with zero auxiliary-state loss weight.

`training.Trainer.evaluate` computes loss, states, neural gradients and initial-state gradients for a chunk. `accumulate` combines window-weighted losses and gradients across a full batch; call `adam` once with the accumulated gradient. Validation passes `backward=False` to use the same forward equations without retaining a backward tape or computing gradients.

The default chunk is 256 windows. Direct calls support 1–2,048 windows, 2–128 samples, 75 modes and up to 1,000 hidden units. Gradients cover the four neural parameter arrays and initial displacement/velocity, not physical coefficients or contact topology. Training uses Accelerate float64 with independent CPU window tiles. Linear projections advance alongside the state, and tiles share a transposed weight matrix for hidden projections.

The [bundled CUDA fixtures](../../tests/fixtures/stable_modal_synthesis/README.md) contain exact inputs, initial model/optimizer state and expected derivatives/updates from author float64 autograd. These are independent reproduction checks; routine code changes should compare against captured results from the accepted local implementation. After building the native targets, run the comparison with:

```sh
script/ValidateStableModal --output build/stable-modal-training-check
```

Use a fresh output directory. The suite checks loss, states, gradients, Adam updates and moments at relative tolerance `1e-6`, plus exact agreement between loss-only and derivative-enabled evaluation, streaming block independence and preservation of explicit auxiliary state. It requires no network, CUDA device, author checkout or dataset. Use `--golden path/to/reference.pt.gz` to check another capture.

## Continuous learned rendering

`render.Stream` snapshots the learned parameters and physical coefficients at construction. It carries displacement, velocity, SAV auxiliary state and the cached neural projection across calls. There is one learned forward implementation, shared with training in `TrainAccelerate.cpp`.

```python
from script.stable_modal_synthesis.render import Stream
from script.stable_modal_synthesis.training import parameters, synthesis_inputs

inputs = synthesis_inputs(model, batch)
with Stream(parameters(model), inputs) as stream:
    for excitation in inputs['excitation'].split(256, dim=1):
        states = stream.process(excitation)  # bodies, 151, frames
```

Each returned sample is the state before consuming its corresponding excitation sample. The stream advances once per input sample, so concatenated blocks have no duplicated boundary state. Initial conditions may contain 150 displacement/velocity values, which initialize the auxiliary state from the potential, or 151 values with explicit auxiliary state. Calls on each stream must be sequential. Parameter changes require a new stream; a physical state alone does not preserve the projection cache exactly.

`render.render(model, batch)` returns complete states and pickup audio in the author format. The native stream allocates its scratch space at construction; scratch memory does not grow with recording length. The Python adapter allocates output tensors per block, and the complete-recording helper retains the whole trajectory. This is a synthesis API, without an audio-device callback or live parameter editing.

## Local data and training

`local_data.py` regenerates trajectories with the unchanged author float64 CPU solver and archived CUDA-drawn parameters, without new random draws. This is a CPU regeneration adaptation, not a byte-identical replacement for CUDA targets. Every normalized waveform must match the released audio within relative L2 `1e-6`. The dataset manifest records this provenance and per-file hashes; loading verifies every checksum.

The default pilot contains eight training recordings and one validation recording, occupying about 1.9 GiB. Dataset preparation requires the pinned author checkout and its released audio. Generate the pilot and check three updates against author autograd, including fresh-process resume:

```sh
python -m script.stable_modal_synthesis.local_data
python -m script.stable_modal_synthesis.local_check \
  --output build/stable-modal-local-check
```

For all 60 training and 20 validation recordings, use the [complete parameter metadata](../../tests/fixtures/stable_modal_synthesis/training-metadata.json). Generated targets occupy about 18.5 GiB and remain outside Git. Generation defaults to eight recordings at a time; `--batch-size` controls memory use.

```sh
python -m script.stable_modal_synthesis.local_data \
  --metadata tests/fixtures/stable_modal_synthesis/training-metadata.json \
  --output data/stable_modal_synthesis/full
python -m script.stable_modal_synthesis.local_train \
  --data data/stable_modal_synthesis/full \
  --output data/stable_modal_synthesis/runs/local --updates 15
```

Use fresh output directories. The loop preserves the author's four-recording update boundaries, epoch shuffle seed 0, 1 ms teacher forcing, mean loss and one Adam step per batch. Fifteen updates complete one full-data epoch; two complete one pilot epoch. Initial model and Adam state come from the bundled initialization fixture. Local runs do not resume the archived 409-epoch CUDA run.

Each update atomically saves `last.pt`. Completed epochs evaluate teacher-forced validation loss and retain `best.pt` when it improves. Resume requires the same dataset, starting fixture, runtime sources and chunk size:

```sh
python -m script.stable_modal_synthesis.local_train \
  --data data/stable_modal_synthesis/full \
  --output data/stable_modal_synthesis/runs/local --updates 15 \
  --resume data/stable_modal_synthesis/runs/local/last.pt
```

`progress.json` and `report.json` record update timing and epoch losses. The report identifies the dataset manifest and its CPU-generation provenance, and separates startup verification, loading, preparation, computation, Adam, validation and checkpoint costs. Teacher-forced validation loss is not a full-trajectory audio-quality measure.

## Regenerating CUDA references

To regenerate a CUDA reference on any SSH-accessible CUDA machine, obtain the pinned public author checkout locally and run:

```sh
python -m script.stable_modal_synthesis.capture_remote \
  --host root@HOST --port PORT \
  --inputs tests/fixtures/stable_modal_synthesis/initial.pt.gz \
           tests/fixtures/stable_modal_synthesis/restored.pt.gz \
           tests/fixtures/stable_modal_synthesis/release-96k.pt.gz \
  --output build/stable-modal-new-cuda-capture
```

This uploads the pinned source bundle and embedded fixture inputs, installs the pinned dependencies over the machine's CUDA PyTorch environment, runs only the retained update counts, and downloads/hash-verifies all captures. It never needs the original RunPod disk or regenerates the 77 GB dataset, and it does not provision or stop the remote machine. `training_reference --inputs ... --device cpu|cuda` provides the same capture locally. `pack_training_fixture` retains exact prefixes with the full-capture hash when replacing a bundled fixture; update its manifest after comparison passes.

## RunPod

Use one H100 80 GB and a 250 GB volume mounted at `/workspace`. The launch command requires 180 GiB free; the original datasets alone occupy approximately 77 GB. Keep the GPU environment and generated results on that volume.

```sh
python3 -m script.stable_modal_synthesis.runpod stage --host root@HOST --port PORT
python3 -m script.stable_modal_synthesis.runpod launch --host root@HOST --port PORT
python3 -m script.stable_modal_synthesis.runpod status --host root@HOST --port PORT
python3 -m script.stable_modal_synthesis.runpod fetch --host root@HOST --port PORT
```

The runner uses SSH and stages only this reproduction harness, then clones the public author reference. Add `--identity PATH` if the SSH agent/default identity is insufficient. It does not provision, resize, stop, or delete paid resources. Results and checkpoints are fetched under `build/stable-modal-cuda` by default.

Before the full run, an optional CUDA execution diagnostic is:

```sh
cd /workspace/stable-modal-synthesis
PYTHONPATH=harness env/bin/python -m script.stable_modal_synthesis.cuda_check \
  --reference reference --output cuda-check.json
PYTHONPATH=harness env/bin/python -m script.stable_modal_synthesis.golden \
  --reference reference --output golden-full --cuda-graph
```

Training checkpoints retain the best and two most recent epochs. Relaunching uses the original trainer's checkpoint resume behavior. An interrupted dataset generation leaves an incomplete author dataset directory; inspect and regenerate that partition before relaunching. Preserve partial results and diagnose failures before removing data.

After preserving the previous attempt's log, status, and harness, use `runpod launch --start-at train` to resume after completed preparation, or `--start-at evaluate` after completed training. Supply the same host and port arguments. Training resumes only after verifying that all split comparisons and the pinned released-checkpoint report are present and passed. These options reuse existing artifacts; they do not regenerate or recertify changed dataset files.

The current run was stopped by the user at 409 epochs; do not restart training without a new request. Final evaluation uses `runpod launch --start-at evaluate --epochs 409`. The final checkpoint filename and evaluation follow this completed epoch count; the pinned author configuration remains unchanged.
