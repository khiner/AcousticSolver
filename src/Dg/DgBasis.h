#pragma once
#include <Eigen/Dense>
#include <array>

namespace dg::reference {
template<class T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Mat = Matrix<double>;
using Vec = Eigen::VectorXd;
struct Basis {
    Mat V;
    std::array<Mat, 3> D;
};
Basis Bernstein(const Mat &, int degree);
struct Quadrature {
    Mat Points;
    Vec Weights;
    std::array<Mat, 4> Faces;
    Vec FaceWeights;
};
Quadrature Integrate(int order);
} // namespace dg::reference
