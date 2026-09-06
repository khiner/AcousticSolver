#!/usr/bin/env python3
"""Build and run upstream native CUDA benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

PIN = "f08c22f83fd64605a634b94326f07cb88f00b0ef"
URL = "https://github.com/dtu-act/libparanumal.git"
HERE = Path(__file__).resolve().parent
CASES = {"cube_rigid": "setup_cube_500hz_freq_indep", "cylinder": "setup_cylinder_250hz_perf_refl"}
SCHEDULES = {"cube_rigid": (656, 2**-15), "cylinder": (65536, 2**-17)}

def command(argv, cwd=None, env=None, log=None):
    print("+", shlex.join(map(str, argv)), flush=True)
    if log:
        with open(log, "w") as out:
            result = subprocess.run(list(map(str, argv)), cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT)
        if result.returncode:
            print("\n".join(Path(log).read_text(errors="replace").splitlines()[-35:]), file=sys.stderr)
            raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    else:
        subprocess.run(list(map(str, argv)), cwd=cwd, env=env, check=True)

def output(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def checkout(args):
    if not args.checkout.exists():
        command(["git", "clone", "--depth", "1", URL, args.checkout])
        if output(["git", "rev-parse", "HEAD"], args.checkout) != PIN:
            command(["git", "fetch", "--depth", "1", "origin", PIN], args.checkout)
        command(["git", "checkout", "--detach", PIN], args.checkout)
    if output(["git", "rev-parse", "HEAD"], args.checkout) != PIN:
        raise RuntimeError(f"reference checkout must be pinned at {PIN}: {args.checkout}")
    if output(["git", "status", "--porcelain", "--untracked-files=no"], args.checkout):
        raise RuntimeError("reference checkout has tracked changes; use a clean pinned checkout")

def tree(args, precision):
    return args.work / 'cuda' / precision

def environment(dest):
    env = os.environ.copy()
    occa = dest / "occa"
    env.update(OCCA_DIR=str(occa), OCCA_CACHE_DIR=str(dest / "reference-cache"),
               OCCA_CUDA_ENABLED="1", OCCA_OPENCL_ENABLED="0",
               OCCA_OPENMP_ENABLED="0", OCCA_MPI_ENABLED="0", OCCA_HIP_ENABLED="0")
    env["LD_LIBRARY_PATH"] = str(occa / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = "/usr/local/cuda/bin:" + env["PATH"]
    env["OCCA_INCLUDE_PATH"] = "/usr/local/cuda/include"
    env["OCCA_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs"
    return env

def parse_config(text):
    result = {}
    key = None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            key = line[1:-1]
        elif line and key:
            result[key] = line
            key = None
    return result

def run_one(args, case, precision, label, steps, dt, probe=False):
    dest = tree(args, precision)
    solver = dest / "solvers/acoustics"
    directory = args.output / 'cuda' / case / precision / label
    if directory.exists():
        raise RuntimeError(f"output already exists; use a new work directory: {directory}")
    directory.mkdir(parents=True)
    cfg = parse_config((solver / "tests/setups" / CASES[case]).read_text())
    cfg.update({"OUTPUT DIRECTORY": str(directory / "wav"), "SIMULATION_ID": case,
                "THREAD MODEL": "CUDA", "FINAL TIME": str(steps*dt),
                "WRITE_WAVE_FIELD": "NONE"})
    if case == "cylinder":
        cfg.update({"POLYNOMIAL DEGREE": "6", "MESH FILE": "../../cylinder-p6.msh"})
    assets = {}
    for key in ("MESH FILE", "RECEIVER"):
        if key in cfg:
            path = (solver / cfg[key]).resolve()
            assets[key] = {"upstream_path": str(path.relative_to(dest)), "sha256": sha(path)}
            cfg[key] = str(path)
    if case == "cube_rigid":
        lines = Path(cfg["MESH FILE"]).read_text().splitlines()
        begin = lines.index("$Elements")
        for i in range(begin + 2, lines.index("$EndElements")):
            parts = lines[i].split()
            if parts[1] == "2":  # Gmsh v2 triangle's first tag is its physical surface.
                parts[3] = "1"
                lines[i] = " ".join(parts)
        lines = [line.replace('2 2 "Frequency Independent"', '2 1 "Rigid"') for line in lines]
        mesh = directory / "rigid_cube.msh"
        mesh.write_text("\n".join(lines) + "\n")
        cfg["MESH FILE"] = str(mesh)
        assets["MESH FILE"]["derived_sha256"] = sha(mesh)
        receivers = directory / "receivers.dat"
        receivers.write_text("2\n0.1 0.1 0.1\n0.7 0.2 0.4\n")
        cfg["RECEIVER"] = str(receivers)
        assets["RECEIVER"]["derived_sha256"] = sha(receivers)
    nodes = dest / "nodes" / f"tetN{int(cfg['POLYNOMIAL DEGREE']):02d}.dat"
    assets["REFERENCE NODES"] = {"upstream_path": str(nodes.relative_to(dest)), "sha256": sha(nodes)}
    cfg.update({"REFERENCE DT": str(dt), "REFERENCE STEPS": str(steps),
                "TEMPORAL_PPW_OUTPUT": "1000000000"})
    config_path = directory / "setup.dat"
    config_path.write_text("\n\n".join(f"[{k}]\n{v}" for k, v in cfg.items()) + "\n")
    env = environment(dest)
    env["DG_REFERENCE_OUTPUT"] = str(directory)
    if probe:
        env["DG_REFERENCE_PROBE"] = "1"
    command([solver / "acousticsMain", config_path], solver, env, directory / "run.log")
    meta = json.loads((directory / "solver.json").read_text())
    if meta["backend"] != "CUDA" or meta["scalar_bytes"] != (8 if precision == "fp64" else 4):
        raise RuntimeError("solver backend or precision differs from the requested reference")
    import numpy as np
    scalar = "<f" + str(meta["scalar_bytes"])
    requested_xyz = np.loadtxt(cfg["RECEIVER"], skiprows=1).reshape(-1, 3).astype(scalar)
    actual_xyz = np.fromfile(directory / "receiver_xyz.bin", dtype=scalar).reshape(-1, 3)
    if not np.array_equal(requested_xyz, actual_xyz):
        raise RuntimeError("solver receiver coordinates differ from the input file")
    provenance = {"build": json.loads((dest / "reference-built.json").read_text()), "assets": assets,
                  "config": cfg, "probe": probe,
                  "files": {p.name: sha(p) for p in directory.glob("*.bin")}}
    save(directory / "provenance.json", provenance)
    return meta

def build(args):
    checkout(args)
    patch = HERE/'upstream.patch'
    for precision in ('fp64', 'fp32'):
        dest = tree(args, precision)
        shutil.copytree(args.checkout, dest, ignore=shutil.ignore_patterns('.git', '*.o'))
        command(['git', 'apply', str(patch)], dest)
        if precision == 'fp64':
            types = dest/'include/types.h'
            types.write_text(types.read_text().replace('//float data type\n#if 1', '//float data type\n#if 0', 1))
        # Propagation kernels and the complete RK stage routine must remain upstream.
        original = [args.checkout/'solvers/acoustics/src/acousticsStep.c',
                    *sorted((args.checkout/'solvers/acoustics/okl').rglob('*.okl'))]
        kernels = {str(p.relative_to(args.checkout)): sha(p) for p in original}
        assert all(sha(dest/p) == digest for p, digest in kernels.items())
        makefile = (args.checkout/'solvers/acoustics/makefile').read_text()
        objects = makefile[makefile.index('OBJS    ='):makefile.index('OBJS_MAIN')]
        (dest/'solvers/acoustics/reference.mk').write_text("""CXX ?= mpicxx
CXXFLAGS ?= -O2 -std=c++20
CPPFLAGS += -I. -I../../include -I../../libs/gatherScatter -I$(OCCA_DIR)/include -DOCCA_VERSION_1_0 -DDHOLMES='"$(abspath ../..)"' -DDACOUSTICS='"$(CURDIR)"'
""" + objects + """
acousticsMain: $(OBJS) src/acousticsMain.o
\t$(CXX) $(CXXFLAGS) $^ -L$(OCCA_DIR)/lib -locca $(LDLIBS) -o $@
%.o: %.c
\t$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x c++ -c $< -o $@
""")
        env = environment(dest)
        command(['make', f'-j{args.jobs}', 'CXX=c++', 'CXXFLAGS=-O2'], dest/'occa', env, dest/'occa-build.log')
        includes = output(['pkg-config', '--cflags', 'hdf5'])
        links = output(['pkg-config', '--libs', 'hdf5'])+' -larmadillo -llapack -lblas -ldl -lpthread -lgomp'
        argv = ['make', '-f', 'reference.mk', f'-j{args.jobs}', 'CXX=mpicxx',
                'CXXFLAGS=-O2 -std=c++20 -include pthread.h '+includes, 'LDLIBS='+links]
        command(argv, dest/'solvers/acoustics', env, dest/'solver-build.log')
        save(dest/'reference-built.json', dict(commit=PIN, precision=precision, backend='CUDA',
             patch_sha256=sha(patch), runner_sha256=sha(__file__), kernel_sha256=kernels,
             build_command=argv, compiler=output(['mpicxx','--version']),
             gpu=output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv']),
             cuda=output(['/usr/local/cuda/bin/nvcc','--version']), cuda_math='upstream fast-math retained'))


def cylinder(path):
    import gmsh
    gmsh.initialize()
    for name,value in [('General.Terminal',0),('General.NumThreads',1),('Mesh.RandomSeed',1),
                       ('Mesh.HighOrderFixBoundaryNodes',1),('Mesh.MeshSizeMin',.75),
                       ('Mesh.MeshSizeMax',.75),('Mesh.MshFileVersion',2.2)]:
        gmsh.option.setNumber(name,value)
    gmsh.model.add('rigid_cylinder')
    gmsh.model.occ.addCylinder(0,0,0,0,0,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2,[tag for dim,tag in gmsh.model.getEntities(2)],1)
    gmsh.model.setPhysicalName(2,1,'Rigid')
    gmsh.model.addPhysicalGroup(3,[tag for dim,tag in gmsh.model.getEntities(3)],2)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(6)
    gmsh.model.mesh.optimize('HighOrder')
    gmsh.write(str(path))
    gmsh.finalize()


def cuda(args):
    import numpy as np
    build(args)
    for precision in ('fp64','fp32'):
        shutil.copyfile(args.mesh,tree(args,precision)/'cylinder-p6.msh')
    results = {}
    for case,(steps,dt) in SCHEDULES.items():
        for precision in ('fp64','fp32'):
            run_one(args,case,precision,'mesh',steps,dt,probe=True)
        results[case] = []
        for repeat in range(5):
            label = f'repeat{repeat}' if repeat else 'warm'
            run_one(args,case,'fp32',label,steps if repeat else 64,dt)
            directory = args.output/'cuda'/case/'fp32'/label
            count = steps if repeat else 64
            meta = json.loads((directory/'solver.json').read_text())
            times = np.fromfile(directory/'times.bin',dtype='<f4')
            receivers = np.fromfile(directory/'receivers.bin',dtype='<f4')
            state = np.fromfile(directory/'final_q.bin',dtype='<f4')
            assert np.array_equal(times,np.arange(count)*dt)
            assert receivers.size == count*meta['receivers']
            assert state.size == meta['elements']*4*meta['nodes_per_element']
            assert np.isfinite(receivers).all() and np.isfinite(state).all()
            if repeat > 1:
                for name in ('times.bin','receivers.bin','final_q.bin'):
                    assert (directory/name).read_bytes() == (directory.parent/'repeat1'/name).read_bytes()
            if repeat:
                results[case].append(json.loads((directory/'propagation.json').read_text())['seconds'])
        save(args.output/'timings.json',results)
    print(json.dumps(results,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action',required=True)
    mesh = sub.add_parser('mesh')
    mesh.add_argument('output',type=Path)
    c = sub.add_parser('cuda')
    c.add_argument('--work',type=Path,required=True)
    c.add_argument('--mesh',type=Path,required=True)
    c.add_argument('--jobs',type=int,default=8)
    a = p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path): setattr(a,key,value.resolve())
    if a.action == 'mesh':
        if a.output.exists(): raise RuntimeError('Use a new mesh path')
        cylinder(a.output)
    elif a.action == 'cuda':
        a.work.mkdir(parents=True)
        a.checkout,a.output = a.work/'upstream',a.work/'runs'
        cuda(a)


if __name__ == '__main__':
    main()
