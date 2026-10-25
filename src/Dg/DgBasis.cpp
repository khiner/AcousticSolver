#include "DgBasis.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dg::reference {
namespace {
void Require(bool ok, const char *message) {
    if (!ok) throw std::invalid_argument(message);
}
} // namespace
Basis Bernstein(const Mat &points, int degree) {
    int const n = (degree + 1) * (degree + 2) * (degree + 3) / 6;
    Basis out{Mat(points.rows(), n), {Mat::Zero(points.rows(), n), Mat::Zero(points.rows(), n), Mat::Zero(points.rows(), n)}};
    auto const factorial = [](int k) { double v=1; for(int i=2;i<=k;++i)v*=i; return v; };
    int col = 0;
    for (int i = 0; i <= degree; ++i)
        for (int j = 0; j <= degree - i; ++j)
            for (int k = 0; k <= degree - i - j; ++k, ++col) {
                std::array<int, 4> powers{degree - i - j - k, i, j, k};
                double factor = factorial(degree);
                for (int const power : powers) factor /= factorial(power);
                for (int q = 0; q < points.rows(); ++q) {
                    std::array<double, 4> bary{-(points.row(q).sum() + 1) / 2, (points(q, 0) + 1) / 2, (points(q, 1) + 1) / 2, (points(q, 2) + 1) / 2};
                    auto const product = [&](int lower) { double value=factor; for(int a=0;a<4;++a) value*=std::pow(bary[a],powers[a]-(a==lower)); return value; };
                    out.V(q, col) = product(-1);
                    for (int a = 0; a < 3; ++a) {
                        if (powers[a + 1]) out.D[a](q, col) += .5 * powers[a + 1] * product(a + 1);
                        if (powers[0]) out.D[a](q, col) -= .5 * powers[0] * product(0);
                    }
                }
            }
    return out;
}
Quadrature Integrate(int order) {
    Require(order >= 2 && order <= 64, "Invalid quadrature order");
    Mat jacobi = Mat::Zero(order, order);
    for (int i = 1; i < order; ++i) jacobi(i, i - 1) = jacobi(i - 1, i) = i / std::sqrt(4. * i * i - 1);
    Eigen::SelfAdjointEigenSolver<Mat> const eig(jacobi);
    Require(eig.info() == Eigen::Success, "Gauss quadrature failed");
    Vec g = (eig.eigenvalues().array() + 1) / 2, w = eig.eigenvectors().row(0).array().square();
    Quadrature out;
    int const count = order * order * order;
    out.Points.resize(count, 3);
    out.Weights.resize(count);
    int q = 0;
    for (int i = 0; i < order; ++i)
        for (int j = 0; j < order; ++j)
            for (int k = 0; k < order; ++k, ++q) {
                double a = g[i], b = g[j], c = g[k];
                out.Points.row(q) << 2 * a - 1, 2 * (1 - a) * b - 1, 2 * (1 - a) * (1 - b) * c - 1;
                out.Weights[q] = 8 * w[i] * w[j] * w[k] * (1 - a) * (1 - a) * (1 - b);
            }
    int const nf = 6 * order * order;
    for (auto &f : out.Faces) f.resize(nf, 3);
    out.FaceWeights.resize(nf);
    // Permutation symmetry gives adjacent elements the same face measure.
    std::array<int, 3> permutation{0, 1, 2};
    q = 0;
    do {
        for (int i = 0; i < order; ++i)
            for (int j = 0; j < order; ++j, ++q) {
                std::array<double, 3> bary{1 - g[i] - (1 - g[i]) * g[j], g[i], (1 - g[i]) * g[j]};
                double u = bary[permutation[1]], v = bary[permutation[2]];
                out.Faces[0].row(q) << 2 * u - 1, 2 * v - 1, -1;
                out.Faces[1].row(q) << 2 * u - 1, -1, 2 * v - 1;
                out.Faces[2].row(q) << 2 * u - 1, 2 * v - 1, 1 - 2 * u - 2 * v;
                out.Faces[3].row(q) << -1, 2 * u - 1, 2 * v - 1;
                out.FaceWeights[q] = 4 * w[i] * w[j] * (1 - g[i]) / 6;
            }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return out;
}
} // namespace dg::reference
