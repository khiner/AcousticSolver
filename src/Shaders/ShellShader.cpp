// Ported from WaveBlender (c) 2024 Kangrui Xue (ShellShader.cu) — Metal port.
// Implements the Shell acoustic shader (right now, it's just loading precomputed simulation data from disk).
// The boundary velocities are computed on the CPU (in parallel over boundary points)
// and uploaded directly into the shader data buffer.

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "Shaders.h"

#include "Parallel.h"

void Shell::ReadShellAnimation() {
    auto &base = Obj;
    VertDisplace = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Zero(base.V0.rows(), 3);
    const int lookahead = base.NSamples - 1;

    const int frame1 = base.Step + lookahead; // For now, we assume shader srate = 44.1 kHz and start time = 0 (as in the reference)
    std::stringstream ss;
    ss << std::setw(9) << std::setfill('0') << frame1;

    int n_u = 0, n_c = 0;

    // UNCONSTRAINED
    auto file = ShellAnimDir + ss.str() + ".displacement";
    std::ifstream in_file1{file, std::ios::in | std::ios::binary};
    if (in_file1.good()) {
        in_file1.read(reinterpret_cast<char *>(&n_u), sizeof(int));
        in_file1.read(reinterpret_cast<char *>(VertDisplace.data()), n_u * sizeof(double));

        // CONSTRAINED
        file = ShellAnimDir + ss.str() + ".constraint_displacement";
        std::ifstream in_file2{file, std::ios::in | std::ios::binary};

        in_file2.read(reinterpret_cast<char *>(&n_c), sizeof(int));
        in_file2.read(reinterpret_cast<char *>(VertDisplace.data()) + n_u, n_c * sizeof(double));
    }

    assert(base.V2.rows() == (n_u + n_c) / 3);
    base.V1 = base.V2;
    base.Tree.Init(base.V1, base.F);
    for (int r = 0; r < base.V2.rows(); ++r) {
        const int internal_id = VertMap[r];
        if (internal_id < (n_u + n_c) / 3) base.V2.row(r) = base.V0.row(r) + VertDisplace.row(internal_id).cast<REAL>();
    }
    base.Changed = true;
}

void Shell::ReadVertexMap(const std::string &map_file) {
    VertMap.assign(Obj.V0.rows(), 0);
    std::ifstream in_file{map_file};
    if (in_file.good()) { // if vertex_map file exists, read in
        std::string line;
        std::getline(in_file, line);

        int r = 0;
        while (!line.empty() && in_file.good()) {
            std::istringstream is{line};
            std::getline(in_file, line);
            int orig2intern, intern2orig;
            is >> orig2intern >> intern2orig;
            if (r < int(VertMap.size())) VertMap[r] = orig2intern;
            r += 1;
        }
    } else { // otherwise, use identity mapping
        for (int r = 0; r < int(VertMap.size()); ++r) VertMap[r] = r;
    }
}

void Shell::Compute(GpuBuffer &vb, int global_bid) {
    auto &base = Obj;
    Eigen::VectorXi indices1; // Closest triangle
    base.ClosestPoint(indices1, base.B, base.V1);

    // Load next batch of vertex velocities
    if (base.Step % (MegaBatchSize - 1) == 0) {
        const Eigen::RowVectorXd init_vert_vels = VertVels.row(VertVels.rows() - 1);

        VertVels = Eigen::MatrixXd::Zero(MegaBatchSize, base.V2.rows() * 3);
        for (int k = 0; k < MegaBatchSize; ++k) {
            const int frame = base.Step + k; // For now, assume shader srate = 44.1 kHz (as in the reference)
            std::stringstream ss;
            ss << std::setw(9) << std::setfill('0') << frame;

            int n_u = 0, n_c = 0;

            // UNCONSTRAINED
            auto file = ShellAccelDir + ss.str() + ".wsacc";
            std::ifstream in_file1{file, std::ios::in | std::ios::binary};
            VertAccel0 = VertAccel1;
            if (in_file1.good()) {
                in_file1.read(reinterpret_cast<char *>(&n_u), sizeof(int));
                in_file1.read(reinterpret_cast<char *>(VertAccel1.data()), n_u * sizeof(double));

                // CONSTRAINED
                file = ShellAccelDir + ss.str() + ".constraint_acceleration";
                std::ifstream in_file2{file, std::ios::in | std::ios::binary};

                in_file2.read(reinterpret_cast<char *>(&n_c), sizeof(int));
                in_file2.read(reinterpret_cast<char *>(VertAccel1.data()) + n_u, n_c * sizeof(double));
            } else {
                std::cout << "MISSING Shell accel. file: " << file << std::endl;
                VertAccel1.setZero();
            }
            if (k == 0) VertVels.row(0) = init_vert_vels;
            else VertVels.row(k) = VertVels.row(k - 1) + (VertAccel0 + VertAccel1) / 2. * base.Dt;
        }
    }

    BatchVels = Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(base.NPoints, base.NSamples);
    const int offset = base.Step % (MegaBatchSize - 1);
    ParallelFor(base.NPoints, 64, [&](size_t bid) {
        const Eigen::Vector3<REAL> bn = base.BN.row(bid);
        const Eigen::Vector3<REAL> bn_abs = bn.cwiseAbs();

        for (int k = 0; k < base.NSamples; ++k) {
            for (int vert = 0; vert < 3; ++vert) {
                const REAL weight1 = 1. / 3.; // (1.f - alpha) * weights1(bid, vert);
                const int internal_id1 = VertMap[base.F(indices1[bid], vert)];

                const Eigen::Vector3d vert_vel1 = VertVels.row(offset + k).segment<3>(3 * internal_id1); // assumes shader srate = 44.1 kHz
                BatchVels(bid, k) += weight1 * bn_abs.dot(vert_vel1.cast<REAL>());
            }
        }
    });
    base.Step += base.NSamples - 1;

    // Copy boundary velocity into the device shader data buffer directly
    vb.Upload(BatchVels.data(), BatchVels.size() * sizeof(REAL), size_t(global_bid) * base.NSamples * sizeof(REAL));

    ReadShellAnimation(); // read next set of vertex displacements
}
