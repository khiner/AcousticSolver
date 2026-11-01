# DiffImpact

Apple Silicon implementation of [DiffImpact](https://proceedings.mlr.press/v164/clarke22a.html), with the [complete paper and appendices](https://samuelpclarke.com/files/clarkeCorl2021.pdf) and a pinned author TensorFlow/DDSP reference. MLX runs the differentiable contact, modal body, acceleration, filtered noise, reverb, spectral objective, and audio encoder on Metal in float32. Matrix operations explicitly disable MLX's reduced-precision float32 default.

The released synthesis, fitting, encoder and decoder components are ported. ASMR graph wiring and training settings are explicit reconstruction choices; Appendix D property networks are paper-described reconstructions. Native perceptual metrics and baseline-output comparison are implemented. [Curated ASMR tapping v1](../../config/diffimpact/asmr-v1.json) supplies our own curated recordings and splits for joint audio-encoder/body training. **The full paper experiments are not reproduced:** the original ASMR selection/configuration and property-network checkpoints are absent from the author release, and the advertised Sound-20K soundtrack download is currently inaccessible. External baseline training implementations and the human study are not included.

## Train and render

From the AcousticSolver root, install Python 3.12 dependencies and obtain the pinned recordings:

```sh
python3.12 -m venv build/reference/diffimpact-native-env
build/reference/diffimpact-native-env/bin/pip install -r script/diffimpact/requirements-native.txt
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive external/diffimpact

build/reference/diffimpact-native-env/bin/python -m script.diffimpact.train hammer \
  --output build/diffimpact-hammer
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.train separation \
  --output build/diffimpact-separation
```

Hammer fitting uses the four released recordings, 256 modes per body, and 10,000 joint contact/body updates. Source separation uses the notebook's 32 selected robot recordings, 128 modes per object, 64 per tool, shared object/tool parameters, and 6,000 updates of the entire batch. Both use the released Adam settings and internal 88.2 kHz synthesis. `--steps` changes the total update budget, including already completed updates when resuming.

Outputs require a new directory. Each directory contains the preprocessed recordings, source hashes, model configuration, implementation/dependency provenance, optimizer state, deterministic random-stream seed and position, audio, inferred force, and physical modal parameters. Resuming or evaluating requires only this directory and the native dependencies, not the author checkout or a remote pod:

```sh
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.train hammer \
  --output build/diffimpact-hammer --resume
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.train hammer \
  --output build/diffimpact-hammer --evaluate
```

Interrupting requests a stop after the current update and saves the checkpoint. Hammer and separation runs also export their current outputs. `rendered.npz` includes the mixture, force at the internal sample rate, and separate body, acceleration, noise and room contributions. Source separation also exports the dry tool and the paper's tool-plus-acceleration-and-reverb result. Measured hammer force is evaluation-only; inferred force remains uncalibrated. The reported force error fits a positive scale, not a timing adjustment.

## Audio encoder

The encoder accepts MFCCs derived from audio, predicts time-varying impact magnitude and duration, and jointly learns the shared body and environment. Each object has its own model, as in the paper. Frames from the source video are not model inputs.

### Curated ASMR tapping v1

Our [versioned selection](../../config/diffimpact/asmr-v1.json) uses public ASMR Bakery recordings: [glass bowl, ceramic plate, wooden box and silicone coaster](https://www.youtube.com/watch?v=sIgkTYTWPz8), plus a [metal spoon](https://www.youtube.com/watch?v=FPnfzPozRW4). The spoon replaces the paper's steel bar; silicone is retained as a deformable comparison. These are our selected intervals and splits, not recovered author annotations. Material descriptions are not calibrated physical-property labels.

Each object supplies 80 training, 10 validation and 10 test clips, each three seconds long: 25 minutes of usable audio in total. Entire time intervals are assigned to one split, with three-second gaps. This evaluates held-out tapping sequences within a recording session, not generalization to a different recording setup or object. Object identity was checked against chapter descriptions and sampled video frames; there are no contact-force or per-impact annotations.

Source videos and decoded stereo WAVs live under `data/diffimpact/`, outside disposable build outputs. Their hashes, source timestamps and extraction settings are retained. Fitting uses the left channel; video remains available for later audiovisual work. Keep the local media archives: source websites may change or remove uploads. The media are third-party recordings; our dataset specification does not change their ownership or license.

To acquire the two source archives and build a fresh dataset directory:

```sh
python3.12 -m venv build/reference/media-env
build/reference/media-env/bin/pip install yt-dlp==2026.10.26
build/reference/media-env/bin/yt-dlp --no-playlist -f '136+140' --merge-output-format mp4 \
  --write-info-json -o 'data/diffimpact/sources/%(id)s.%(ext)s' \
  'https://www.youtube.com/watch?v=sIgkTYTWPz8' 'https://www.youtube.com/watch?v=FPnfzPozRW4'
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.curate
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.train autoencoder \
  --manifest data/diffimpact/asmr-v1/glass-bowl.json --steps 10000 \
  --output build/diffimpact-asmr-glass
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.train autoencoder \
  --output build/diffimpact-asmr-glass --evaluate
```

`curate` requires FFmpeg/ffprobe, verifies the pinned source archives, decodes exact sample intervals, and validates every training manifest. It rejects existing output directories, overlapping intervals and missing guard gaps. The 10,000-step command is a chosen training budget, not an author ASMR configuration. The dataset's adaptation labels propagate into training checkpoints and result reports. These recordings do not replace synthetic Sound-20K training data for Appendix D.

### Encoder configuration and custom recordings

The public release contains `MfccTimeDistributedRnnEncoder` and `RnnFcDecoder`, but no ASMR clips, split manifest, training configuration or trained encoder. The native implementation follows those classes: instance-normalized MFCCs → GRU → temporal convolution with leaky ReLU → linear 32-channel projection → latent resampling → three dense/normalization/leaky-ReLU blocks → GRU with a concatenated skip connection → three more blocks → linear contact outputs. The projection is linear in the released source, although Appendix C describes an activation there. Each reset-after GRU retains Keras's two bias rows as one trainable tensor, preserving both gradients and per-variable Adam clipping.

Widths use DDSP defaults: 512 recurrent/dense channels, kernel 7 and leaky ReLU slope .2. Connecting only the latent `z` to magnitude/duration outputs, using 100 Hz controls and disabling stochastic contact perturbation are explicit ASMR wiring choices. The shared physical decoder defaults and clipped Adam settings come from the hammer setup, with the paper's additional 128-sample spectral-loss window. These settings are not a recovered author ASMR configuration.

Supply a manifest for **one object**, with disjoint training, validation and test intervals. `start` and `stop` are integer sample indices in the source WAV; `channel` selects a stereo channel. Recordings must be 44.1 kHz. Intervals are divided into non-overlapping clips of `samples` samples; incomplete tails are discarded. Each clip is mean-centered and peak-normalized. Hash mismatches and overlapping source intervals are rejected.

```json
{
  "object": "steel-bar",
  "rate": 44100,
  "samples": 132300,
  "recordings": [
    {"path": "steel.wav", "sha256": "...", "split": "train", "start": 0, "stop": 8820000, "channel": 0},
    {"path": "steel.wav", "sha256": "...", "split": "validation", "start": 8820000, "stop": 9702000, "channel": 0},
    {"path": "steel.wav", "sha256": "...", "split": "test", "start": 9702000, "stop": 10584000, "channel": 0}
  ]
}
```

```sh
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.train autoencoder \
  --manifest /path/to/steel-bar.json --steps 10000 --output build/diffimpact-steel-bar
```

The step budget above is an example, not an author schedule. Training draws deterministic minibatches only from the training split. It evaluates validation spectral loss at initialization and every checkpoint interval, using fixed noise, and retains its minimum as `best.npz`. `checkpoint.npz` is the latest resumable state; `initial.npz` retains the untrained baseline. The validation interval is saved when creating the run, and its history and best step persist across resumes. Training does not render or score the test split.

After selecting a model using validation, run `train autoencoder --output ... --evaluate`. This loads `best.npz`, renders each split, and records both the selected step and the total completed training steps. A larger total `--steps` with `--resume` continues from `checkpoint.npz`. Validation-based selection is our protocol choice; the supplied splits and preprocessing are also adaptations. The paper's ASMR comparisons, CDPAM/human-study results, and Sound-20K material-classification results have not been reproduced by this port. Source separation alone does not establish the material-classification result.

Apply a trained object's encoder to another recording of that object:

```sh
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.infer \
  --checkpoint build/diffimpact-steel-bar/best.npz \
  --audio /path/to/new-steel-strikes.wav --output build/diffimpact-inferred
```

Inference exports audio, component stems, inferred force, magnitude/duration controls and physical modal coefficients. It normalizes each bounded clip independently and pads the last clip; it is not continuous-state streaming. Output records the normalization and source/checkpoint hashes.

## Material and stiffness inference

`properties.py` implements Appendix D's GRU material classifier and dilated-convolution Young's-modulus regressor. Training uses only synthetic Sound-20K tracks, with validation-selected `best.npz` and resumable `checkpoint.npz`. The classifier uses equal counts of ceramic, polycarb, steel and wood; the regressor uses all available materials and mean squared logarithmic error in Pascals. Both use grouped, seeded 90/10 splits. Groups cannot cross the training/validation boundary.

The paper does not provide widths, dropout rates, exact spectrogram preprocessing or optimizer settings. The reconstruction uses 128 channels, 128/64 dense layers, dropout .2, a 512-sample Hann spectrogram with hop 128, mean centering and peak normalization, and the released fitting code's clipped Adam. The regressor uses kernel 3 with dilations 1/2/4, masked global max pooling and an exponential log-Pascal output head. These choices, the grouped split and early-stopping patience are adaptations, recorded in each checkpoint and [protocol.json](protocol.json).

Obtain the original Sound-20K soundtrack archive and the pinned [SoundSynth metadata](https://github.com/ztzhang/SoundSynth/tree/db6ddd9f36de7ea8b19c83a40aa268338be48365). The indexer accepts the generator's `scene-*/obj-*/mat-*` directory layout and numbered isolated-object WAV tracks; it excludes the fixed floor and never labels merged mixtures as isolated objects. A different archive layout requires an explicit manifest with `dataset: "Sound-20K"` and `recordings` entries containing `path`, `sha256`, `group`, `material` and `youngs_pa`.

```sh
git submodule update --init external/soundsynth
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.sound20k \
  --audio /path/to/sound-20k --output build/sound20k.json
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.properties material \
  --manifest build/sound20k.json --epochs 200 --output build/material
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.properties youngs \
  --manifest build/sound20k.json --epochs 200 --output build/youngs
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.evaluate robot \
  --run build/diffimpact-separation \
  --material build/material/best.npz --youngs build/youngs/best.npz \
  --output build/robot-properties.json
```

The epoch limit is a chosen ceiling, not an author schedule. Default patience is ten validation epochs. `--resume --epochs N` uses the saved minibatch position, dropout stream, optimizer and cached features without requiring the original audio. The robot evaluator uses the released recording labels, with nominal modulus values from the synthetic material configurations, not measured stiffness of the real utensils. It reports named-source confusion matrices and accuracies, plus Table 1's Dataset/Instance Max accuracy and Table 2's Dataset/Instance Min percentage error for both object and tool. Accuracies are fractions in `[0,1]`.

## Audio metrics and baseline comparisons

The frozen [CDPAM](https://github.com/pranaymanocha/PerceptualAudio) inference network runs natively in MLX. Bootstrap its public, hash-pinned pretrained weights once in a separate conversion environment:

```sh
python3.12 -m venv build/reference/cdpam-env
build/reference/cdpam-env/bin/pip install torch==2.8.0 numpy==1.26.4
build/reference/cdpam-env/bin/python -m script.diffimpact.capture_metric
```

Conversion also captures a one-time independent CPU reference. Normal evaluation needs only `build/reference/cdpam/weights.npz` and the native dependencies. The weights are reproducible from the pinned public package; they do not depend on a pod or a previous training run. CDPAM preserves its 22.05 kHz, rounded PCM-scale input convention; using SciPy polyphase resampling in place of the reference loader's librosa resampling is an adaptation.

`evaluate audio` computes CDPAM, six-scale linear/log spectral distance (2048 through 64 samples), and RMS Hilbert-envelope distance. Each metric is normalized by the corresponding **dataset mean** distance to seeded uniform `[-1,1]` white noise. The paper does not specify the noise distribution or level, so these normalized scores are not an exact reconstruction of its plotted numbers.

An audio evaluation manifest references NPZ arrays by path, key and SHA-256. Each array is `[recordings, samples]`, in matching order, at 44.1 kHz. The evaluator does not align, trim or normalize predictions:

```json
{
  "rate": 44100,
  "target": {"path": "dataset.npz", "key": "test.audio", "sha256": "..."},
  "predictions": {
    "DiffImpact": {"path": "rendered.npz", "key": "test.audio", "sha256": "..."},
    "external-baseline": {"path": "baseline.npz", "key": "audio", "sha256": "..."}
  }
}
```

```sh
build/reference/diffimpact-native-env/bin/python -m script.diffimpact.evaluate audio \
  --manifest /path/to/evaluation.json --output build/audio-metrics.json
```

Robot reports retain their complete manifest. Save that `manifest` object to JSON and add methods under `predictions`, each with `object` and `tool` array references, then use `evaluate robot --manifest ... --output ...` to compare external NMF/DAP outputs. For an unordered baseline those two names simply designate its first and second outputs; dataset/instance extrema handle source assignment. This imports matched predictions; it does not implement or train WaveGlow, DDSP comparison models or DAP, and supplies no human-study results.

## Numerical boundary

The native implementation preserves the notebook parameter transforms, convolution alignment, noise and reverb filtering, nonnegative constraints, and per-variable clipped legacy Adam. The hammer notebook uses 2048/1024/512/256 spectral windows with log weight 1 and fixed reverb parameters. The separation notebook adds 128, uses log weight .1, and learns tool-specific reverb. These released-code choices differ from a single universal paper configuration.

Three native adaptations are deliberate:

- Correct contact indexing within each recording. The released `Impact.get_signal` gathers batch-local indices from flattened tensors without batch offsets, coupling different recordings. The independent reference check evaluates each recording separately to verify the correction.
- Evaluate constant-frequency oscillator phase directly instead of accumulating it in float32. This preserves the equation and one-sample initial phase while avoiding cumulative phase drift. Long-response checks use an independent float64 analytic kernel.
- Use MLX initialization and random streams. Identical seeds do not imply identical TensorFlow/MLX draws or optimization trajectories. Flat contact envelopes have nondifferentiable peak ties; floating-point changes can select different peaks.

Checkpoints preserve parameters, Adam moments and random-stream position exactly. Repeated native GPU runs can still differ numerically; continuation is checked with floating-point tolerances rather than a promise of bit-identical trajectories.

See [validation coverage](../../VALIDATION.md#diffimpact) for the accepted local fixtures and source-comparison limits. Run the regression suite with:

```sh
script/ValidateDiffImpact
```

After bootstrapping the frozen metric weights, append `--cdpam-weights build/reference/cdpam/weights.npz` to check CDPAM against its accepted local fixture. No repeated author-framework comparison is needed.

The fixture directory also retains the final native 10,000-step hammer and 6,000-step separation checkpoints, optimizer states and render signatures. These are adapted native fits, not author checkpoints. Regression checks exercise both trained states without downloading recordings or using a remote machine. The source recordings and full rendered outputs remain outside Git.

`capture.py` and `capture_encoder.py` capture independent author comparisons when mathematical coverage changes. They are not part of normal regression checks. To run the unchanged author hammer workflow, install `requirements.txt` into `build/reference/diffimpact-env` with `setuptools==80.9.0`, then run:

```sh
build/reference/diffimpact-env/bin/python -m script.diffimpact.run --output build/diffimpact-author
build/reference/diffimpact-env/bin/python -m script.diffimpact.check --output build/diffimpact-author-check.json
```

The reference uses TensorFlow 2.16.2/legacy Keras on Apple Silicon rather than the author's TensorFlow 2.4.2 Linux environment. [protocol.json](protocol.json) pins source/data hashes; [NOTICE](NOTICE) retains the upstream licenses. Numerical port validation and completion of a notebook schedule do not constitute reproduction of every paper experiment.
