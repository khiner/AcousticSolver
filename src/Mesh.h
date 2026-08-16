#pragma once

#include <Eigen/Core>

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

// Mean curvature (the average of the two principal curvatures) at every vertex, from a
// quadric fit over its 5-ring neighborhood in the tangent frame, as in the reference
// implementation. Vertices with too small a neighborhood to fit get zero.
Eigen::VectorXd MeanCurvature(const Eigen::MatrixXd &v, const Eigen::MatrixXi &f);
