#!/usr/bin/env python3
"""Run every WaveBlender scene and collect its CUDA reference listener outputs.

Executed on the GPU host from WaveBlender's build/ directory (uploaded there by
generate_golden_outputs.sh). Outputs land in build/cuda/<scene>/.
"""
import json, os, shutil, subprocess, sys, time

SCENES_DIR = "../Scenes"
OUT_DIR = "cuda"


def write_wav(bin_path, srate):
    """Peak-normalized .bin -> .wav, as in the reference write_wav.py (without its matplotlib import)."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        print(f"WAV CONVERSION SKIPPED ({e}) — install numpy + soundfile", flush=True)
        return
    data = np.fromfile(bin_path, dtype=np.float32)
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak
    sf.write(os.path.splitext(bin_path)[0] + ".wav", data, srate)

results = []
for scene in sorted(os.listdir(SCENES_DIR)):
    config_path = os.path.join(SCENES_DIR, scene, "config.json")
    if not os.path.isfile(config_path):
        continue
    with open(config_path) as f:
        config = json.load(f)
    srate = config["FDTD_srate"]
    outputs = [l["output"] for l in config["listeners"].values()]

    print(f"=== {scene} (srate={srate}, tf={config['tf']}) ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(["./WaveBlender", config_path])
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"EXIT {proc.returncode}"
    results.append((scene, status, elapsed))
    print(f"=== {scene}: {status} in {elapsed:.1f}s ===", flush=True)
    if proc.returncode != 0:
        continue

    dest = os.path.join(OUT_DIR, scene)
    os.makedirs(dest, exist_ok=True)
    for name in outputs:
        bin_path = name + ".bin"
        if os.path.isfile(bin_path):
            write_wav(bin_path, srate)
            for ext in (".bin", ".wav"):
                if os.path.isfile(name + ext):
                    shutil.move(name + ext, os.path.join(dest, name + ext))
        else:
            print(f"MISSING OUTPUT: {bin_path}", flush=True)

print("\n===== SUMMARY =====")
for scene, status, elapsed in results:
    print(f"{scene:15s} {status:8s} {elapsed:8.1f}s")
