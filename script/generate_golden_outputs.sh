#!/usr/bin/env bash
# Generate golden CUDA reference outputs for all WaveBlender scenes on a remote
# NVIDIA GPU host (e.g. a RunPod RTX 4090 pod) and fetch them back.
#
# Usage:
#   script/generate_golden_outputs.sh <user@host> <ssh_port> [identity_file]
#
# Example (RunPod "SSH over exposed TCP" endpoint):
#   script/generate_golden_outputs.sh root@213.173.108.152 13450 ~/.ssh/id_rsa
#
# Requirements on the remote host: Ubuntu-ish with CUDA toolkit at
# /usr/local/cuda (any RunPod CUDA/PyTorch template works), root apt access.
# The run streams over one SSH session (ServerAliveInterval keeps it up) and
# takes on the order of an hour for all 10 scenes.
#
# Layout assumptions (override via env vars):
#   WAVEBLENDER_DIR  CUDA reference repo      (default: <this repo>/../WaveBlender)
#   FLUIDSOUND_DIR   FluidSound clone         (default: <this repo>/../FluidSound)
#   GOLDEN_DIR       local output destination (default: <this repo>/golden)
set -euo pipefail

HOST=${1:?usage: generate_golden_outputs.sh <user@host> <ssh_port> [identity_file]}
PORT=${2:?usage: generate_golden_outputs.sh <user@host> <ssh_port> [identity_file]}
KEY=${3:-$HOME/.ssh/id_rsa}

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WAVEBLENDER_DIR=${WAVEBLENDER_DIR:-$REPO_ROOT/../WaveBlender}
FLUIDSOUND_DIR=${FLUIDSOUND_DIR:-$REPO_ROOT/../FluidSound}
GOLDEN_DIR=${GOLDEN_DIR:-$REPO_ROOT/golden}

SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
          -o ServerAliveInterval=30 -p "$PORT")
run() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
RSYNC_RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p $PORT"

SCENES="CupPhone 2016Pour GlassPour PaddleSplash LegoDrop SpollingBowl TalkFan Trumpet HandShake FillerUp"

echo "==> Installing dependencies on $HOST"
run 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq >/dev/null && apt-get install -y -qq libeigen3-dev unzip rsync >/dev/null'
run 'python3 -m pip install --quiet --break-system-packages numpy soundfile 2>/dev/null || python3 -m pip install --quiet numpy soundfile' # for run_golden.py's .bin -> .wav conversion

echo "==> Uploading WaveBlender source ($WAVEBLENDER_DIR) and FluidSound ($FLUIDSOUND_DIR)"
rsync -az --exclude '.git' --exclude 'Scenes' --exclude 'build' -e "$RSYNC_RSH" "$WAVEBLENDER_DIR"/ "$HOST":'~/WaveBlender/'
rsync -az --exclude '.git' -e "$RSYNC_RSH" "$FLUIDSOUND_DIR"/ "$HOST":'~/WaveBlender/Source/FluidSound/'
rsync -az -e "$RSYNC_RSH" "$REPO_ROOT/script/run_golden.py" "$HOST":'~/WaveBlender/'

echo "==> Fetching scene data from the Stanford dataset (skips scenes already present)"
run 'cd ~/WaveBlender && mkdir -p Scenes && cd Scenes && for f in '"$SCENES"'; do
       [ -d "$f" ] && { echo "have $f"; continue; }
       curl -sSfLO "https://graphics.stanford.edu/papers/waveblender/dataset/data/$f.zip" \
         && unzip -qo "$f.zip" && rm "$f.zip" && echo "fetched $f" || echo "FAILED: $f"
     done'

echo "==> Building"
run 'export PATH=/usr/local/cuda/bin:$PATH && cd ~/WaveBlender && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. > cmake.log && make -j"$(nproc)" 2>&1 | tail -3'

echo "==> Running all scenes (streams progress; this takes a while)"
run 'cd ~/WaveBlender/build && cp ../run_golden.py . && python3 run_golden.py 2>&1 | tee golden_run.log'

echo "==> Fetching outputs to $GOLDEN_DIR"
mkdir -p "$GOLDEN_DIR"
rsync -az -e "$RSYNC_RSH" "$HOST":'~/WaveBlender/build/golden/' "$GOLDEN_DIR"/
rsync -az -e "$RSYNC_RSH" "$HOST":'~/WaveBlender/build/golden_run.log' "$GOLDEN_DIR"/

echo "==> Done. Golden outputs in $GOLDEN_DIR"
