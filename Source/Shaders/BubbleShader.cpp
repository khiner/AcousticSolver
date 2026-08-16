// Ported from WaveBlender (c) 2024 Kangrui Xue (BubbleShader.cu) — Metal port.
// Implements the Bubbles acoustic shader for bubble-based water sound
// (kernels: bub_to_boundary, bub_flux, bub_matmul in Kernels.metal).
//
// GpuBubData and GpuBubVels hold one slot per minibatch: each minibatch's upload lands
// in its own slot (bound via buffer offset), so uploads never overwrite data that
// already-encoded, not-yet-executed kernels read — the whole batch encodes without a
// single GPU sync.
//
// References:
//   [Xue et al. 2023] Improved Water Sound Synthesis using Coupled Bubbles
//   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains

#include <iomanip>
#include <iostream>
#include <sstream>

#include "Shaders.h"

#include "KernelParams.h"

void Bubbles::ReadFluidMesh() {
    auto &base = Obj;
    const int lookahead = base.NSamples - 1;

    // Bunch of filename logic
    std::string filetag = "tmpMesh-00";
    filetag += std::to_string((base.Step + lookahead + int(base.Srate * base.Ts)) / base.Srate) + ".";

    const int dec = (base.Step + lookahead + int(base.Srate * base.Ts)) % base.Srate / (base.Srate / 1000);
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << dec;
    filetag += ss.str() + "000";

    std::cout << "Reading mesh from " << FluidMeshDir + filetag << std::endl;
    base.V1 = base.V2;
    base.Changed = ReadObj(FluidMeshDir + filetag + ".obj", base.V2, base.F); // true if ReadObj() successful, false otherwise
}

// As in the reference: probably will not work on batch lengths that are non-integer multiples of 1 ms.
void Bubbles::Compute(GpuBuffer &vb, int global_bid) {
    auto &ctx = MetalContext::Get();
    auto &base = Obj;

    const int minibatchsize = std::min(base.Srate / 1000, base.NSamples - 1); // 1 ms
    const int n_minibatches = ((base.NSamples - 1) + (minibatchsize - 1)) / minibatchsize; // ceiling

    // Determine active oscillators during current batch
    ActiveOscIds.clear();
    const double t1 = base.Step * base.Dt + base.Ts;
    const double t2 = (base.Step + base.NSamples - 1) * base.Dt + base.Ts;
    for (int os_id = 0; os_id < int(Solver.oscillators().size()); ++os_id) {
        const auto &osc = Solver.oscillators()[os_id];

        if (osc.startTime <= t1 && osc.endTime > t1) ActiveOscIds.push_back(os_id); // coupled oscillators
        else if (osc.endTime <= t1 && !osc.is_dead()) ActiveOscIds.push_back(os_id); // uncoupled oscillators
        else if (t1 < osc.startTime && osc.startTime < t2) ActiveOscIds.push_back(os_id); // osc will be added during current batch
    }

    // Initialize a bunch of buffers
    const int n_bubs = ActiveOscIds.size();
    BubData.resize(n_bubs * 4);
    BubVels = Eigen::MatrixX<REAL>::Zero(n_bubs, minibatchsize + 1);

    const Dim3 threads{16, 16, 1};
    const Dim3 blocks{uint32_t(base.NPoints + 15) / 16, uint32_t(n_bubs + 15) / 16, 1};
    const Dim3 matmul_blocks{uint32_t(base.NPoints + 15) / 16, uint32_t((minibatchsize + 1) + 15) / 16, 1};

    const size_t data_slot_bytes = BubData.size() * sizeof(REAL);
    const size_t vels_slot_bytes = BubVels.size() * sizeof(REAL);
    GpuBubData.Resize(n_minibatches * data_slot_bytes);
    GpuBubVels.Resize(n_minibatches * vels_slot_bytes);
    Flux.Resize(n_bubs * sizeof(REAL));
    BubToBoundary.Resize(size_t(base.NPoints) * n_bubs * sizeof(REAL));

    for (int m = 0; m < n_minibatches; ++m) {
        // Construct bubble-to-boundary projection matrix (NPoints x n_bubs)
        for (int j = 0; j < n_bubs; ++j) {
            auto &osc = Solver.oscillators()[ActiveOscIds[j]];

            if ((base.Step * base.Dt + base.Ts) > osc.endTime && osc.is_dead()) continue;
            const Eigen::ArrayXd data = osc.interp(base.Step * base.Dt + base.Ts);
            BubData[4 * j] = data[0]; // r
            BubData[4 * j + 1] = data[2]; // xbub
            BubData[4 * j + 2] = data[3]; // ybub
            BubData[4 * j + 3] = data[4]; // zbub
        }
        GpuBubData.Upload(BubData.data(), data_slot_bytes, m * data_slot_bytes);
        const BubToBoundaryParams btb_params{base.NPoints, n_bubs};
        ctx.Dispatch("bub_to_boundary", blocks, threads, {&BubToBoundary, &base.GpuB, &base.GpuBN, {&GpuBubData, m * data_slot_bytes}}, &btb_params, sizeof(btb_params));

        // Per-bubble boundary flux, alpha = cell face area (replaces the reference's cublasSgemv)
        const BubFluxParams flux_params{base.NPoints, n_bubs, Dx * Dx};
        ctx.Dispatch("bub_flux", {uint32_t(n_bubs + 31) / 32, 1, 1}, {32, 1, 1}, {&BubToBoundary, &Flux}, &flux_params, sizeof(flux_params));

        // Package oscillator velocities (over minibatch) into matrix (n_bubs x T_steps)
        const int start = m * minibatchsize;
        int stop = (m + 1) * minibatchsize;
        for (int k = start; k < stop; ++k) {
            for (int j = 0; j < n_bubs; ++j) {
                const auto &osc = Solver.oscillators()[ActiveOscIds[j]];
                BubVels(j, k % minibatchsize) = osc.state[1];
            }
            Solver.step();
            base.Step += 1;
        }

        // Extend last minibatch for interpolation
        if (m == n_minibatches - 1) {
            for (int j = 0; j < n_bubs; ++j) {
                const auto &osc = Solver.oscillators()[ActiveOscIds[j]];
                BubVels(j, minibatchsize) = osc.state[1];
            }
            stop += 1;
        }
        // Matrix multiply to get boundary velocity
        GpuBubVels.Upload(BubVels.data(), vels_slot_bytes, m * vels_slot_bytes);
        const BubMatmulParams mm_params{global_bid, base.NPoints, n_bubs, base.NSamples, start, stop};
        ctx.Dispatch("bub_matmul", matmul_blocks, threads, {&vb, &BubToBoundary, {&GpuBubVels, m * vels_slot_bytes}, &Flux}, &mm_params, sizeof(mm_params));
    }

    ReadFluidMesh(); // read next mesh
}
