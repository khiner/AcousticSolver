// Ported from WaveBlender (c) 2024 Kangrui Xue (Object.cpp) — Metal port.
// ObjectBase geometry, animation, and closest-point implementation.

#include "Parallel.h"
#include "Shaders.h"

#include "igl/barycentric_coordinates.h"
#include "igl/read_triangle_mesh.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {
// Fast path for plain triangle meshes ("v x y z" and "f a b c" lines, as the per-batch
// fluid meshes are). strtod/strtol are the same correctly-rounded conversions igl's
// sscanf-based reader uses, so values are identical. Returns false, leaving the outputs
// untouched, on any feature outside that subset.
bool ReadObjFast(const std::string &filename, Eigen::MatrixX<REAL> &v_out, Eigen::MatrixXi &f_out) {
    std::ifstream in{filename, std::ios::binary | std::ios::ate};
    if (!in.good()) return false;
    const auto size = size_t(in.tellg());
    in.seekg(0);
    std::string data(size, '\0');
    in.read(data.data(), size);

    std::vector<REAL> vs;
    std::vector<int> fs;
    const char *p = data.data();
    const char *const end = p + size;
    const auto skip_line = [&] {
        while (p < end && *p != '\n') ++p;
        if (p < end) ++p;
    };
    const auto at_line_end = [&] {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) ++p;
        return p == end || *p == '\n';
    };
    while (p < end) {
        if (*p == 'v' && p + 1 < end && p[1] == ' ') {
            p += 2;
            for (int c = 0; c < 3; ++c) {
                char *q{};
                const double value = std::strtod(p, &q);
                if (q == p) return false;
                vs.push_back(REAL(value));
                p = q;
            }
            if (!at_line_end()) return false; // vertex colors / 4-component positions
            skip_line();
        } else if (*p == 'f' && p + 1 < end && p[1] == ' ') {
            p += 2;
            for (int c = 0; c < 3; ++c) {
                char *q{};
                const long index = std::strtol(p, &q, 10);
                if (q == p || index <= 0) return false; // relative/invalid indices
                fs.push_back(int(index - 1));
                p = q;
                if (p < end && *p == '/') return false; // texture/normal face indices
            }
            if (!at_line_end()) return false; // quads and larger polygons
            skip_line();
        } else if (*p == '#' || *p == '\n' || *p == '\r' || *p == ' ' || *p == '\t' || (*p == 'v' && p + 1 < end && (p[1] == 'n' || p[1] == 't')) || *p == 'g' || *p == 's' || *p == 'o') {
            skip_line(); // comments, blank lines, normals/texcoords, grouping
        } else {
            return false; // anything else (mtllib, lines, ...) takes the general path
        }
    }

    v_out = Eigen::Map<const Eigen::Matrix<REAL, Eigen::Dynamic, 3, Eigen::RowMajor>>(vs.data(), long(vs.size()) / 3, 3);
    f_out = Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>(fs.data(), long(fs.size()) / 3, 3);
    return true;
}
} // namespace

bool ObjectBase::ReadObj(const std::string &filename, Eigen::MatrixX<REAL> &v, Eigen::MatrixXi &f) {
    if (ReadObjFast(filename, v, f)) return true;
    return igl::read_triangle_mesh(filename, v, f);
}

void ObjectBase::SetSamplePoints(const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &bn) {
    NPoints = b.rows();
    B = b;
    BN = bn;

    GpuB.Resize(B.size() * sizeof(REAL));
    GpuBN.Resize(BN.size() * sizeof(REAL));
    GpuB.Upload(B.data(), B.size() * sizeof(REAL));
    GpuBN.Upload(BN.data(), BN.size() * sizeof(REAL));
}

void ObjectBase::ReadAnimation() {
    const int lookahead = NSamples - 1;
    std::string line;

    if (Step == 0) std::getline(AnimFile, line); // First line is metadata

    while (AnimFile.good() && !AnimFile.eof() && T2 <= (Step + lookahead) * Dt + Ts) {
        std::getline(AnimFile, line);
        T1 = T2;
        Translation1 = Translation2;
        Rotation1 = Rotation2;

        std::istringstream is{line};
        is >> T2 >> Translation2[0] >> Translation2[1] >> Translation2[2] >> Rotation2[0] >> Rotation2[1] >> Rotation2[2] >> Rotation2[3];
    }
    if (Step == 0) {
        Translation1 = Translation2;
        Rotation1 = Rotation2;
    }

    // Same fixed threshold as the reference (see its TODO)
    if (Step > 0 && (Translation2 - Translation1).norm() < 1e-6 && (Rotation2 - Rotation1).norm() < 1e-6) {
        Changed = false;
        return;
    }
    // Scale Rot Trans
    const Eigen::Quaterniond quat1{Rotation1[0], Rotation1[1], Rotation1[2], Rotation1[3]};
    const Eigen::Quaterniond quat2{Rotation2[0], Rotation2[1], Rotation2[2], Rotation2[3]};

    const double alpha = ((Step + lookahead) * Dt + Ts - T1) / (T2 - T1);
    const Eigen::Quaternion<REAL> quat = quat1.slerp(alpha, quat2).cast<REAL>();
    const Eigen::Vector3<REAL> translation = ((1. - alpha) * Translation1 + alpha * Translation2).cast<REAL>();

    V1 = V2;
    for (int r = 0; r < V2.rows(); ++r) {
        V2.row(r) = quat * V0.row(r);
        V2.row(r) += translation;
    }
    Changed = true;
}

void ObjectBase::ClosestPoint(Eigen::VectorXi &i_out, const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &v) const {
    i_out.resize(b.rows());
    ParallelChunks(b.rows(), 256, [&](size_t begin, size_t end) {
        Eigen::VectorX<REAL> sqr_d;
        Eigen::MatrixX<REAL> p;
        Eigen::VectorXi indices;
        const Eigen::MatrixX<REAL> block = b.middleRows(begin, end - begin);
        Tree.squared_distance(v, F, block, sqr_d, indices, p);
        i_out.segment(begin, end - begin) = indices;
    });
}

void ObjectBase::ClosestPoint(Eigen::VectorXi &i_out, Eigen::MatrixX<REAL> &w_out, const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &v) const {
    i_out.resize(b.rows());
    w_out.resize(b.rows(), 3);
    ParallelChunks(b.rows(), 256, [&](size_t begin, size_t end) {
        const auto n = end - begin;
        Eigen::VectorX<REAL> sqr_d;
        Eigen::MatrixX<REAL> p;
        Eigen::VectorXi indices;
        const Eigen::MatrixX<REAL> block = b.middleRows(begin, n);
        Tree.squared_distance(v, F, block, sqr_d, indices, p);

        Eigen::MatrixX<REAL> tri1(n, 3), tri2(n, 3), tri3(n, 3);
        for (size_t r = 0; r < n; ++r) {
            tri1.row(r) = v.row(F(indices[r], 0));
            tri2.row(r) = v.row(F(indices[r], 1));
            tri3.row(r) = v.row(F(indices[r], 2));
        }
        Eigen::MatrixX<REAL> weights;
        igl::barycentric_coordinates(p, tri1, tri2, tri3, weights);

        i_out.segment(begin, n) = indices;
        w_out.middleRows(begin, n) = weights;
    });
}
