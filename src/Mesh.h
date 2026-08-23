#pragma once

#include <Eigen/Core>

#include <cmath>
#include <string>
#include <vector>

// Reads a Wavefront .obj into flat, row-major positions and vertex indices, ignoring every
// line but `v` and `f`. Returns false if the file cannot be opened or a `v`/`f` line is
// malformed.
bool ReadObjData(const std::string &filename, std::vector<double> &vs, std::vector<int> &fs);

// As above, into an (#V x 3) vertex matrix and an (#F x 3) face matrix.
template<typename DerivedV, typename DerivedF> bool ReadObj(const std::string &filename, DerivedV &v, DerivedF &f) {
    std::vector<double> vs;
    std::vector<int> fs;
    if (!ReadObjData(filename, vs, fs)) return false;

    v.resize(long(vs.size()) / 3, 3);
    for (long r = 0; r < v.rows(); ++r) {
        for (int c = 0; c < 3; ++c) v(r, c) = typename DerivedV::Scalar(vs[r * 3 + c]);
    }
    f.resize(long(fs.size()) / 3, 3);
    for (long r = 0; r < f.rows(); ++r) {
        for (int c = 0; c < 3; ++c) f(r, c) = fs[r * 3 + c];
    }
    return true;
}

// Barycentric weights of `p` within triangle (a, b, c), as a row vector. Both solvers use
// this on an AabbTree closest point, to blend per-vertex data at a point on a face.
// A degenerate triangle falls back to the first vertex rather than dividing by zero.
template<typename Dp, typename Da, typename Db, typename Dc>
Eigen::RowVector3<typename Dp::Scalar> BarycentricWeights(const Eigen::MatrixBase<Dp> &p, const Eigen::MatrixBase<Da> &a, const Eigen::MatrixBase<Db> &b, const Eigen::MatrixBase<Dc> &c) {
    using T = typename Dp::Scalar;
    const Eigen::RowVector3<T> v0 = b - a, v1 = c - a, v2 = p - a;
    const T d00 = v0.dot(v0), d01 = v0.dot(v1), d11 = v1.dot(v1), d20 = v2.dot(v0), d21 = v2.dot(v1);
    const T denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < T(1e-30)) return {T(1), T(0), T(0)};
    const T w1 = (d11 * d20 - d01 * d21) / denom, w2 = (d00 * d21 - d01 * d20) / denom;
    return {T(1) - (w1 + w2), w1, w2};
}

// Mean curvature (the average of the two principal curvatures) at every vertex, from a
// quadric fit over its 5-ring neighborhood in the tangent frame, as in the reference
// implementation. Vertices with too small a neighborhood to fit get zero.
Eigen::VectorXd MeanCurvature(const Eigen::MatrixXd &v, const Eigen::MatrixXi &f);
