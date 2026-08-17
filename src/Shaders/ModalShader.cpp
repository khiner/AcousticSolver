// Ported from WaveBlender (c) 2024 Kangrui Xue (ModalShader.cu) — Metal port.
// Implements the Modal acoustic shader (kernel: ModeMatmul in Kernels.metal).
//
// The impact-record scan is windowed per batch: candidates are collected once, then the
// exact per-sample check runs on that subset in record order (order affects the
// acceleration sum's rounding). Mode-to-boundary rows accumulate in parallel.

#include "Shaders.h"

#include "KernelParams.h"
#include "Parallel.h"
#include "Profile.h"

#include <algorithm>
#include <numbers>
#include <numeric>

namespace {
// Bit equality, deliberately stricter than float equality (reuse must be exact).
bool RowBitsEqual(const Eigen::Matrix<real, Eigen::Dynamic, 3, Eigen::RowMajor> &a, Eigen::Index ra, const Eigen::Matrix<real, Eigen::Dynamic, 3, Eigen::RowMajor> &b, Eigen::Index rb) {
    return std::memcmp(a.data() + 3 * ra, b.data() + 3 * rb, 3 * sizeof(real)) == 0; // NOLINT(bugprone-suspicious-memory-comparison)
}
} // namespace

bool Modal::SetModeToBoundary(Eigen::MatrixX<real> &mode_to_boundary, bool reuse_previous) {
    const profile::Scope scope{"modal/mode_to_boundary"};
    const auto &base = Obj;
    const auto n_modes = mode_to_boundary.cols();
    const Eigen::Quaterniond quat1{base.Rotation1[0], base.Rotation1[1], base.Rotation1[2], base.Rotation1[3]};
    const Eigen::Quaterniond quat2{base.Rotation2[0], base.Rotation2[1], base.Rotation2[2], base.Rotation2[3]};

    // Compute interpolated rotation + translation
    const double alpha = ((base.Step * base.Dt + base.Ts) - base.T1) / (base.T2 - base.T1);
    const Eigen::Quaternion<real> quat = quat1.slerp(alpha, quat2).cast<real>();
    const Eigen::RowVector3<real> translation = ((1. - alpha) * base.Translation1 + alpha * base.Translation2).cast<real>();

    // For closest point query, first transform boundary face positions into rest frame
    const Eigen::Matrix3<real> inv_rot = quat.inverse().toRotationMatrix();
    Eigen::Matrix<real, Eigen::Dynamic, 3, Eigen::RowMajor> b = base.B.rowwise() - translation;
    b = b.eval() * inv_rot.transpose();

    // Match rows against the snapshot (face key, query point, and normal must all
    // bit-match), listing hits for copying and the rest for fresh computation.
    std::vector<int> fresh;
    HitBids.clear();
    HitRows.clear();
    if (reuse_previous) {
        fresh.reserve(base.NPoints);
        for (int bid = 0; bid < base.NPoints; ++bid) {
            const int key = base.SampleKeys[bid];
            const int row = key >= 0 && key < int(KeyToRow.size()) ? KeyToRow[key] : -1;
            if (row >= 0 && RowBitsEqual(b, bid, PrevQ, row) && RowBitsEqual(base.BN, bid, PrevBN, row)) {
                HitBids.push_back(bid);
                HitRows.push_back(row);
            } else {
                fresh.push_back(bid);
            }
        }
        // Every row hit at its own position — leave the matrix unpopulated for the
        // caller to reuse the previous upload
        if (fresh.empty() && int(HitBids.size()) == int(ModeToBoundary2.rows()) &&
            std::ranges::equal(HitBids, HitRows)) {
            return true;
        }
        // Copy hits column by column — per-row copies of column-major matrices would
        // stride the full matrix.
        ParallelChunks(n_modes, 16, [&](size_t begin, size_t end) {
            for (size_t j = begin; j < end; ++j) {
                real *dst = mode_to_boundary.col(j).data();
                const real *src = ModeToBoundary2.col(j).data();
                for (size_t h = 0; h < HitBids.size(); ++h) dst[HitBids[h]] = src[HitRows[h]];
            }
        });
    } else {
        fresh.resize(base.NPoints);
        std::iota(fresh.begin(), fresh.end(), 0);
    }

    Eigen::VectorXi indices; // Closest triangle indices + barycentric weights
    Eigen::MatrixX<real> weights;
    Eigen::Matrix<real, Eigen::Dynamic, 3, Eigen::RowMajor> b_fresh(fresh.size(), 3);
    for (size_t i = 0; i < fresh.size(); ++i) b_fresh.row(i) = b.row(fresh[i]);
    base.ClosestPoint(indices, weights, b_fresh, base.V0);

    // Each fresh point's three weighted eigenvector rows accumulate into a contiguous
    // scratch row (the matrix is column-major — the GPU kernel wants points contiguous —
    // so accumulating in place would stride every element access), then scratch rows
    // move into the matrix in one blocked transpose sweep.
    RowScratch.resize(fresh.size(), n_modes);
    ParallelChunks(fresh.size(), 64, [&](size_t begin, size_t end) {
        Eigen::RowVectorX<real> acc(EvnReal.cols());
        for (size_t i = begin; i < end; ++i) {
            const int bid = fresh[i];
            const Eigen::Vector3<real> bn = base.BN.row(bid);
            acc.setZero();
            for (int vert = 0; vert < 3; ++vert) {
                real weight = weights(i, vert);
                const int vertex_id = base.F(indices[i], vert);

                const Eigen::Vector3<real> normal = NormalsReal.row(vertex_id);
                weight *= bn.dot(quat * normal);

                acc += weight * EvnReal.row(vertex_id);
            }
            RowScratch.row(i) = acc;
        }
    });
    constexpr size_t ModeBlock{16}; // wide enough to reuse scratch cache lines per sweep
    ParallelChunks((n_modes + ModeBlock - 1) / ModeBlock, 1, [&](size_t begin, size_t end) {
        for (size_t blk = begin; blk < end; ++blk) {
            const size_t j0 = blk * ModeBlock, j1 = std::min(j0 + ModeBlock, size_t(n_modes));
            for (size_t i = 0; i < fresh.size(); ++i) {
                const real *src = RowScratch.data() + i * n_modes;
                real *dst = mode_to_boundary.data() + fresh[i];
                for (size_t j = j0; j < j1; ++j) dst[j * size_t(mode_to_boundary.rows())] = src[j];
            }
        }
    });

    // A populating call snapshots its inputs for the next batch's reusing call.
    if (!reuse_previous) {
        for (const int key : PrevKeys) {
            if (key >= 0) KeyToRow[key] = -1;
        }
        PrevKeys = base.SampleKeys;
        int max_key = -1;
        for (const int key : PrevKeys) max_key = std::max(max_key, key);
        if (max_key >= int(KeyToRow.size())) KeyToRow.resize(max_key + 1, -1);
        for (int bid = 0; bid < base.NPoints; ++bid) {
            if (PrevKeys[bid] >= 0) KeyToRow[PrevKeys[bid]] = bid;
        }
        PrevQ = std::move(b);
        PrevBN = base.BN;
    }
    return false;
}

void Modal::Compute(GpuBuffer &vb, int global_bid) {
    auto &base = Obj;
    const int n_modes = Solver.QDotCPlus.size();
    const Eigen::Quaterniond quat1{base.Rotation1[0], base.Rotation1[1], base.Rotation1[2], base.Rotation1[3]};
    const Eigen::Quaterniond quat2{base.Rotation2[0], base.Rotation2[1], base.Rotation2[2], base.Rotation2[3]};

    // Build mode-to-boundary matrix at batch start (we will eventually interpolate in
    // between batch start and end). Every row is written, so size without zeroing. On a
    // full identity hit the matrix is bitwise the previous ModeToBoundary2, so rebind
    // that upload instead — neither buffer is in flight after the previous batch's sync.
    ModeToBoundary1.resize(base.NPoints, n_modes);
    if (SetModeToBoundary(ModeToBoundary1, true)) {
        GpuModeToBoundary1.Swap(GpuModeToBoundary2);
    } else {
        GpuModeToBoundary1.Resize(ModeToBoundary1.size() * sizeof(real));
        GpuModeToBoundary1.Upload(ModeToBoundary1.data(), ModeToBoundary1.size() * sizeof(real));
    }

    ModeVels.setZero(n_modes, base.NSamples);
    AccelNoise.setZero(base.NPoints, base.NSamples);

    // Resolve each point's first nonzero boundary-normal component once for the batch,
    // instead of re-testing all three BN components for every point at every sample.
    std::vector<int8_t> bn_axis(base.NPoints, -1);
    for (int bid = 0; bid < base.NPoints; ++bid) {
        if (base.BN(bid, 0) != 0.f) bn_axis[bid] = 0;
        else if (base.BN(bid, 1) != 0.f) bn_axis[bid] = 1;
        else if (base.BN(bid, 2) != 0.f) bn_axis[bid] = 2;
    }

    // Records whose support can overlap this batch (the exact per-sample check below
    // selects from this subset, preserving record order).
    const auto &impulses = Solver.Impulses.Records;
    const double batch_t1 = base.Step * base.Dt + base.Ts;
    const double batch_t2 = (base.Step + base.NSamples - 1) * base.Dt + base.Ts;
    std::vector<const ModalSound::ImpactRecord *> batch_records;
    for (const auto &record : impulses) {
        if (record.Timestamp + record.SupportLength > batch_t1 && record.Timestamp <= batch_t2) batch_records.push_back(&record);
    }

    Eigen::Vector3<double> accel_pulse1; // current acceleration
    Eigen::Vector3<double> accel_pulse2 = Eigen::Vector3<double>::Zero(); // next acceleration

    // Timestep modal vibrations
    {
        const profile::Scope sample_scope{"modal/sample_loop"};
        for (int k = 0; k < base.NSamples; ++k) {
            ModeVels.col(k) = Solver.QDotCPlus.cast<real>();

            // ----- Acceleration noise -----
            const double time = base.Step * base.Dt + base.Ts;

            const double alpha = (time - base.T1) / (base.T2 - base.T1);
            const Eigen::Quaterniond quat = quat1.slerp(alpha, quat2);

            accel_pulse1 = accel_pulse2;
            accel_pulse2 = Eigen::Vector3<double>::Zero();

            for (const auto *record : batch_records) {
                const auto &impulse = *record;
                if (!(time >= impulse.Timestamp && time < impulse.Timestamp + impulse.SupportLength)) continue;
                if (impulse.SupportLength < 1e-12) continue;

                const Eigen::Vector3d &j_vec = impulse.ImpactVector;
                const Eigen::Vector3d r = impulse.ImpactPosition - Solver.CenterOfMass;

                double s = 0.;
                if (time <= impulse.Timestamp + impulse.SupportLength && time >= impulse.Timestamp) s = std::sin(std::numbers::pi * (time - impulse.Timestamp) / impulse.SupportLength);

                // Translational acceleration
                accel_pulse2 += j_vec * (std::numbers::pi * s / (2. * impulse.SupportLength * Solver.Mass));

                // Rotational acceleration (Eq. 13) from [Chadwick et al. 2012]
                const Eigen::Matrix3d &i_inv = Solver.InvInertia;
                const Eigen::Vector3d rot_alpha = (i_inv * r.cross(j_vec)) * (std::numbers::pi * s) / (2. * impulse.SupportLength);
                accel_pulse2 += rot_alpha.cross(r);
            }

            accel_pulse2 = quat * accel_pulse2.eval(); // rotate vector from rest frame to animation frame

            for (int bid = 0; bid < base.NPoints; ++bid) { // set normal velocities
                if (bn_axis[bid] >= 0) AccelNoise(bid, k) = AccelSum[bn_axis[bid]];
            }

            Solver.Step(base.Step * base.Dt + base.Ts);
            if (k >= base.NSamples - 1) break;

            AccelSum += (accel_pulse1 + accel_pulse2) / 2.f * base.Dt; // trapezoidal rule integrator
            base.Step += 1;
        }
    }

    GpuModeVels.Resize(ModeVels.size() * sizeof(real));
    GpuModeVels.Upload(ModeVels.data(), ModeVels.size() * sizeof(real));

    GpuAccelNoise.Resize(AccelNoise.size() * sizeof(real));
    GpuAccelNoise.Upload(AccelNoise.data(), AccelNoise.size() * sizeof(real));

    // Read next animation data in order to build mode-to-boundary matrix at batch end
    if (base.AnimFile.is_open()) base.ReadAnimation();

    ModeToBoundary2.resize(base.NPoints, n_modes); // every row is written by the populating call
    SetModeToBoundary(ModeToBoundary2, false);

    GpuModeToBoundary2.Resize(ModeToBoundary2.size() * sizeof(real));
    GpuModeToBoundary2.Upload(ModeToBoundary2.data(), ModeToBoundary2.size() * sizeof(real));

    // Multiply mode-to-boundary matrix with modal velocities on GPU
    const Dim3 threads{16, 16, 1};
    const Dim3 matmul_blocks{uint32_t(base.NPoints + 15) / 16, uint32_t(base.NSamples + 15) / 16, 1};

    const ModeMatmulParams params{global_bid, base.NPoints, n_modes, base.NSamples};
    MetalContext::Get().Dispatch("ModeMatmul", matmul_blocks, threads, {&vb, &GpuModeToBoundary1, &GpuModeToBoundary2, &GpuModeVels, &GpuAccelNoise}, &params, sizeof(params));
}
