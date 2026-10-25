"""Native solver invocation and result encoding shared by the command-line tools."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXECUTABLE = ROOT/'build/ThermoviscousSolve'
REVISION = '938ee0630a7a6516c15dea33998472bdbdcbbfc0'
PAPER = 'https://doi.org/10.1142/S2591728523500159'


def executable_hash():
    return hashlib.sha256(EXECUTABLE.read_bytes()).hexdigest()


def complex_values(values):
    a = np.asarray(values)
    return a[:,0]+1j*a[:,1]


def run(scene, directory, *, output='result.json', export=None, quiet=False):
    directory.mkdir(parents=True,exist_ok=True)
    config, result = directory/'scene.json',directory/output
    config.write_text(json.dumps(scene)+'\n')
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS','2')
    subprocess.run([str(EXECUTABLE),str(config),str(result)]+([f'--{export}'] if export else []),
                   env=env,check=True,stdout=subprocess.DEVNULL if quiet else None)
    return json.loads(result.read_text())
