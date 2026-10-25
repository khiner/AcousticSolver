#!/usr/bin/env python3
"""Build the pinned FP64 SeisSol reference on Apple Silicon with Homebrew MPI and LLVM."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = {
    "yaml-cpp": ("https://github.com/jbeder/yaml-cpp.git", "f7320141120f720aecc4c32be25586e7da9eb978"),
    "easi": ("https://github.com/SeisSol/easi.git", "f668a64afa61191d8c9ae4e6adf47aecd2529e00"),
    "hdf5": ("https://github.com/HDFGroup/hdf5.git", "7bf340440909d468dbb3cf41f0ea0d87f5050cea"),
}


def run(output, jobs):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / "prefix"

    def command(args, name):
        with (output / f"{name}.log").open("a") as log:
            subprocess.run(list(map(str, args)), stdout=log, stderr=subprocess.STDOUT, check=True)

    def brew(name):
        return Path(subprocess.check_output(["brew", "--prefix", name], text=True).strip())

    def build(name, source, flags, install=False):
        directory = output / f"{name}-build"
        command(
            ["cmake", "--fresh", "-S", source, "-B", directory, "-G", "Ninja"] + flags,
            name,
        )
        command(["cmake", "--build", directory, "-j", jobs] + (["--target", "install"] if install else []), name)
        return directory

    llvm, mpi = brew("llvm"), brew("open-mpi")
    for name, (url, revision) in DEPENDENCIES.items():
        source = output / name
        if not source.exists():
            command(["git", "init", source], name)
            command(["git", "-C", source, "fetch", "--depth", "1", url, revision], name)
            command(["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"], name)
        actual = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"{source} has unexpected revision {actual}")
        flags = {
            "yaml-cpp": [
                "-DYAML_CPP_BUILD_TESTS=OFF",
                "-DYAML_CPP_BUILD_TOOLS=OFF",
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            ],
            "easi": ["-DASAGI=OFF", "-DIMPALAJIT=OFF", "-DLUA=OFF"],
            "hdf5": [
                f"-DCMAKE_C_COMPILER={mpi}/bin/mpicc",
                "-DHDF5_ENABLE_PARALLEL=ON",
                "-DHDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16=OFF",
                "-DHDF5_BUILD_CPP_LIB=OFF",
                "-DHDF5_BUILD_EXAMPLES=OFF",
                "-DHDF5_BUILD_TOOLS=OFF",
                "-DBUILD_TESTING=OFF",
            ],
        }[name]
        build(
            name,
            source,
            [
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
                f"-DCMAKE_PREFIX_PATH={prefix}",
                "-DCMAKE_BUILD_TYPE=Release",
            ]
            + flags,
            install=True,
        )
    source = ROOT / "external/seissol"
    command(["git", "-C", source, "submodule", "update", "--init", "--recursive"], "seissol-submodules")
    directory = build(
        "seissol",
        source,
        [
            f"-DCMAKE_C_COMPILER={llvm}/bin/clang",
            f"-DCMAKE_CXX_COMPILER={llvm}/bin/clang++",
            f"-DCMAKE_PREFIX_PATH={prefix};{mpi.parent.parent}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DHDF5_ROOT={prefix}",
            f"-DHDF5_DIR={prefix}/cmake",
            f"-DMPI_C_HEADER_DIR={mpi}/include",
            f"-DMPI_CXX_HEADER_DIR={mpi}/include",
            "-DHOST_ARCH=apple-m1",
            "-DGEMM_TOOLS_LIST=Eigen",
            "-DGRAPH_PARTITIONING_LIBS=none",
            "-DPRECISION=double",
            "-DORDER=3",
            "-DEQUATIONS=elastic",
            "-DASAGI=OFF",
            "-DNUMA_AWARE_PINNING=OFF",
            "-DNETCDF=OFF",
        ],
    )
    (output / "build-provenance.json").write_text(
        json.dumps({"dependencies": DEPENDENCIES, "python": sys.version}, indent=2) + "\n"
    )
    print(directory / "seissol-elastic-o3-f64")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/elastic-reference")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.jobs)
