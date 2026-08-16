// Ported from WaveBlender (c) 2024 Kangrui Xue (MonopoleShader.cu) — Metal port.
// Implements the Monopole acoustic shader for testing (kernel: monopole in Kernels.metal).

#include "Shaders.h"

#include "KernelParams.h"

void Monopole::Compute(GpuBuffer &vb, int global_bid) {
    const Dim3 threads{16, 16, 1};
    const Dim3 blocks{uint32_t(Obj.NPoints + 15) / 16, uint32_t(Obj.NSamples + 15) / 16, 1};

    const MonopoleParams params{global_bid, FreqHz, Speed, C, Obj.Srate, Obj.NPoints, Obj.NSamples, Obj.Step};
    MetalContext::Get().Dispatch("monopole", blocks, threads, {&vb, &Obj.GpuB, &Obj.GpuBN}, &params, sizeof(params));
    Obj.Step += Obj.NSamples - 1;

    Obj.V1 = Obj.V2;
    Obj.V2.rowwise() += Eigen::RowVector3<REAL>{Speed, 0., 0.};

    if (Obj.Step > 0 && Speed == 0.) Obj.Changed = false;
}
