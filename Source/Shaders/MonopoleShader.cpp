// Ported from WaveBlender (c) 2024 Kangrui Xue (MonopoleShader.cu) — Metal port.
// Implements the Monopole acoustic shader for testing (kernel: monopole in Kernels.metal).

#include "Shaders.h"

void Monopole::Compute(GpuBuffer &vb, int global_bid) {
    const Dim3 threads{16, 16, 1};
    const Dim3 blocks{uint32_t(NPoints + 15) / 16, uint32_t(NSamples + 15) / 16, 1};

    const MonopoleParams params{global_bid, FreqHz, Speed, 343.2f, Srate, NPoints, NSamples, Step};
    MetalContext::Get().Dispatch("monopole", blocks, threads, {&vb, &GpuB, &GpuBN}, &params, sizeof(params));
    Step += NSamples - 1;

    V1 = V2;
    V2.rowwise() += Eigen::RowVector3<REAL>{Speed, 0., 0.};

    if (Step > 0 && Speed == 0.) Changed = false;
}
