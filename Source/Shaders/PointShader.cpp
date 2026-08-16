// Ported from WaveBlender (c) 2024 Kangrui Xue (PointShader.cu) — Metal port.
// Implements the PointSource force acoustic shader for impulse-related "clicks"
// (kernels: impulse_to_boundary, impulse_matmul in Kernels.metal).
//
// The active-impulse query is windowed: impulses are pre-sorted by start time, so each
// batch scans only those whose [start, start + MaxTau] interval can overlap it before
// the exact per-sample check.

#include <algorithm>
#include <iostream>
#include <numbers>
#include <sstream>

#include "Shaders.h"

#include "KernelParams.h"

void PointSource::ReadImpulses(const std::string &filename) {
    std::ifstream in_file{filename};
    std::string line;
    std::getline(in_file, line); // Skip first line
    while (!line.empty() && in_file.good()) {
        std::getline(in_file, line);
        std::istringstream is{line};

        double time, impulse, tau, dvx, dvy, dvz, px, py, pz;
        int id;
        is >> time >> impulse >> tau >> dvx >> dvy >> dvz >> px >> py >> pz >> id;

        Times.push_back(time);
        Tau.push_back(tau);
        DV.emplace_back(dvx, dvy, dvz);
        Positions.emplace_back(px, py, pz);
    }
    std::cout << "No. impulses: " << Times.size() << std::endl;

    IdsByTime.resize(Times.size());
    for (int idx = 0; idx < int(Times.size()); ++idx) IdsByTime[idx] = idx;
    std::sort(IdsByTime.begin(), IdsByTime.end(), [this](int a, int b) { return Times[a] < Times[b]; });
    MaxTau = Tau.empty() ? 0. : *std::max_element(Tau.begin(), Tau.end());
}

void PointSource::GetActiveImpulseList() {
    auto &base = Obj;
    ActiveImpulseIds.clear();

    // Impulses overlapping the batch start no later than the last sample and no earlier
    // than MaxTau before the first (the exact per-sample check below trims the rest).
    const double t_first = base.Step * base.Dt;
    const double t_last = (base.Step + base.NSamples - 1) * base.Dt;
    const auto begin = std::lower_bound(IdsByTime.begin(), IdsByTime.end(), t_first - MaxTau, [this](int idx, double t) { return Times[idx] < t; });
    const auto end = std::upper_bound(begin, IdsByTime.end(), t_last, [this](double t, int idx) { return t < Times[idx]; });

    for (auto it = begin; it != end; ++it) {
        const int idx = *it;
        for (int k = 0; k < base.NSamples; ++k) {
            const double time = (base.Step + k) * base.Dt;
            if (time >= Times[idx] && time < Times[idx] + Tau[idx]) {
                ActiveImpulseIds.push_back(idx);
                break;
            }
        }
    }
    std::sort(ActiveImpulseIds.begin(), ActiveImpulseIds.end()); // ascending index order fixes row order (and thus rounding)

    base.V2 = Eigen::MatrixX<REAL>::Zero(ActiveImpulseIds.size(), 3);
    int r = 0;
    for (const int idx : ActiveImpulseIds) {
        base.V2.row(r) = Positions[idx];
        r += 1;
    }
    base.Changed = true;
}

void PointSource::Compute(GpuBuffer &force, int global_bid) {
    auto &ctx = MetalContext::Get();
    auto &base = Obj;

    const int n_impulses = ActiveImpulseIds.size();
    const Dim3 threads{16, 16, 1};
    const Dim3 blocks{uint32_t(base.NPoints + 15) / 16, uint32_t(n_impulses + 15) / 16, 1};
    const Dim3 matmul_blocks{uint32_t(base.NPoints + 15) / 16, uint32_t(base.NSamples + 15) / 16, 1};

    ImpulseData.resize(n_impulses * 6);
    ImpulseVels = Eigen::MatrixX<REAL>::Zero(n_impulses, base.NSamples);
    int j = 0;
    for (const int idx : ActiveImpulseIds) {
        const auto &j_vec = DV[idx];
        const auto &pos = Positions[idx];
        const Eigen::Vector3<REAL> dir = j_vec / j_vec.norm();
        for (int k = 0; k < base.NSamples; ++k) {
            const double time = (base.Step + k) * base.Dt;
            if (time < Times[idx] || time > Times[idx] + Tau[idx]) continue;

            const REAL s = std::sin(std::numbers::pi * (time - Times[idx]) / Tau[idx]);

            // translational acceleration
            const Eigen::Vector3<REAL> accel_pulse = std::numbers::pi / (2. * Tau[idx]) * j_vec * s;
            ImpulseVels(j, k) = accel_pulse.norm();
        }
        ImpulseData[6 * j] = pos[0];
        ImpulseData[6 * j + 1] = pos[1];
        ImpulseData[6 * j + 2] = pos[2]; // positions
        ImpulseData[6 * j + 3] = dir[0];
        ImpulseData[6 * j + 4] = dir[1];
        ImpulseData[6 * j + 5] = dir[2]; // directions

        j += 1;
    }
    base.Step += base.NSamples - 1;

    GpuImpulseData.Resize(n_impulses * 6 * sizeof(REAL));
    ImpulseToBoundary.Resize(size_t(n_impulses) * base.NPoints * sizeof(REAL));
    GpuImpulseVels.Resize(ImpulseVels.size() * sizeof(REAL));

    GpuImpulseData.Upload(ImpulseData.data(), ImpulseData.size() * sizeof(REAL));
    const ImpulseToBoundaryParams itb_params{base.NPoints, n_impulses, Dx};
    ctx.Dispatch("impulse_to_boundary", blocks, threads, {&ImpulseToBoundary, &base.GpuB, &base.GpuBN, &GpuImpulseData}, &itb_params, sizeof(itb_params));

    GpuImpulseVels.Upload(ImpulseVels.data(), ImpulseVels.size() * sizeof(REAL));
    const ImpulseMatmulParams mm_params{global_bid, base.NPoints, n_impulses, base.NSamples};
    ctx.Dispatch("impulse_matmul", matmul_blocks, threads, {&force, &ImpulseToBoundary, &GpuImpulseVels}, &mm_params, sizeof(mm_params));

    GetActiveImpulseList();
}
