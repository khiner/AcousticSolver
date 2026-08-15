// Ported from WaveBlender (c) 2024 Kangrui Xue (BubbleShader.cu) — Metal port.
// Implements the Bubbles acoustic shader for bubble-based water sound
// (kernels: bub_to_boundary, bub_flux, bub_matmul in Kernels.metal).
//
// References:
//   [Xue et al. 2023] Improved Water Sound Synthesis using Coupled Bubbles
//   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains

#include "Shaders.h"

void Bubbles::ReadFluidMesh() {
    const int lookahead = NSamples - 1;

    // Bunch of filename logic
    std::string filetag = "tmpMesh-00";
    filetag += std::to_string((Step + lookahead + int(Srate * Ts)) / Srate) + ".";

    const int dec = (Step + lookahead + int(Srate * Ts)) % Srate / (Srate / 1000);
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << dec;
    filetag += ss.str() + "000";

    std::cout << "Reading mesh from " << FluidMeshDir + filetag << std::endl;
    V1 = V2;
    Changed = ReadObj(FluidMeshDir + filetag + ".obj", V2, F); // true if ReadObj() successful, false otherwise
}

// As in the reference: probably will not work on batch lengths that are non-integer multiples of 1 ms.
void Bubbles::Compute(GpuBuffer &vb, int global_bid) {
    auto &ctx = MetalContext::Get();

    const int minibatchsize = std::min(Srate / 1000, NSamples - 1); // 1 ms
    const int n_minibatches = ((NSamples - 1) + (minibatchsize - 1)) / minibatchsize; // ceiling

    // Determine active oscillators during current batch
    ActiveOscIds.clear();
    const double t1 = Step * Dt + Ts;
    const double t2 = (Step + NSamples - 1) * Dt + Ts;
    for (int os_id = 0; os_id < Solver.oscillators().size(); ++os_id) {
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
    const Dim3 blocks{uint32_t(NPoints + 15) / 16, uint32_t(n_bubs + 15) / 16, 1};
    const Dim3 matmul_blocks{uint32_t(NPoints + 15) / 16, uint32_t((minibatchsize + 1) + 15) / 16, 1};

    GpuBubData.Resize(BubData.size() * sizeof(REAL));
    GpuBubVels.Resize(BubVels.size() * sizeof(REAL));
    Flux.Resize(n_bubs * sizeof(REAL));
    BubToBoundary.Resize(size_t(NPoints) * n_bubs * sizeof(REAL));

    for (int m = 0; m < n_minibatches; ++m) {
        // Construct bubble-to-boundary projection matrix (NPoints x n_bubs)
        for (int j = 0; j < n_bubs; ++j) {
            auto &osc = Solver.oscillators()[ActiveOscIds[j]];

            if ((Step * Dt + Ts) > osc.endTime && osc.is_dead()) continue;
            const Eigen::ArrayXd data = osc.interp(Step * Dt + Ts);
            BubData[4 * j] = data[0]; // r
            BubData[4 * j + 1] = data[2]; // xbub
            BubData[4 * j + 2] = data[3]; // ybub
            BubData[4 * j + 3] = data[4]; // zbub
        }
        GpuBubData.Upload(BubData.data(), BubData.size() * sizeof(REAL));
        const BubToBoundaryParams btb_params{NPoints, n_bubs};
        ctx.Dispatch("bub_to_boundary", blocks, threads, {&BubToBoundary, &GpuB, &GpuBN, &GpuBubData}, &btb_params, sizeof(btb_params));

        // Per-bubble boundary flux, alpha = cell face area (replaces the reference's cublasSgemv)
        const BubFluxParams flux_params{NPoints, n_bubs, Dx * Dx};
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
            Step += 1;
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
        GpuBubVels.Upload(BubVels.data(), BubVels.size() * sizeof(REAL));
        const BubMatmulParams mm_params{global_bid, NPoints, n_bubs, NSamples, start, stop};
        ctx.Dispatch("bub_matmul", matmul_blocks, threads, {&vb, &BubToBoundary, &GpuBubVels, &Flux}, &mm_params, sizeof(mm_params));
    }

    ReadFluidMesh(); // read next mesh
}
