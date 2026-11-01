# Stable modal training fixtures

These compressed PyTorch artifacts contain the complete inputs and expected float64 CUDA results for offline native-training regression tests. They do not depend on a RunPod instance, its disk, the original 77 GB dataset, or an author checkout at test time.

Run from the repository root after building the native libraries with `script/Build`:

```sh
script/ValidateStableModal --output build/stable-modal-training-check
```

The suite verifies each artifact against `manifest.json`, then compares loss, trajectories, neural-parameter gradients, initial-state gradients, Adam parameter updates, and both optimizer moments. The fixtures contain four 1 ms windows from recording 0 at window indices 0, 1, 20, and 1000:

- `initial.pt.gz`: two updates from the author's seeded initialization at 88.2 kHz.
- `restored.pt.gz`: two updates from our epoch 409 checkpoint, preserving Adam state at step 6135.
- `release-96k.pt.gz`: one diagnostic update of the released checkpoint with fresh Adam on 96 kHz test-recording windows. It is not used to train or evaluate a production model.

Reference computation used PyTorch 2.8.0 CUDA float64 on an RTX 4090 and the unchanged author solver/loss/Adam, pinned at `68017c953319636b7b69018d8e406f17e18cf0de`. The saved inputs include prepared mode shapes and excitation so the comparison isolates solver and optimizer arithmetic. Full source hashes, environment, model selection, input provenance, and the hashes of longer original captures are embedded in the artifacts. The manifest records the retained sequence lengths and hashes.

The author reference and released model are from [Victor Zheleznov's jaes-modal-node](https://github.com/victorzheleznov/jaes-modal-node). Its MIT license is retained in [LICENSE.reference](LICENSE.reference). See the [reproduction README](../../../script/stable_modal_synthesis/README.md) for fidelity limits and the portable CUDA recapture command. Replacing these fixtures requires independent reference capture and comparison, not generating expected values with the native training implementation.

`pilot-metadata.json` retains the original CUDA-drawn parameters for eight training recordings and one validation recording, plus hashes of the complete archived metadata files. It contains no generated trajectories or optimizer reference outputs. The local pilot rebuilds full trajectories with the pinned author CPU solver and checks them against released audio; see the reproduction README for the explicit adaptation and storage boundary.

`training-metadata.json` retains all 60 training and 20 validation parameter records from the same archived CUDA metadata sources. Pass it to `local_data --metadata` with a fresh output directory to regenerate the complete training/validation splits. Generation still uses CPU float64 and must pass the per-recording released-audio checks; generated trajectories remain outside Git.
