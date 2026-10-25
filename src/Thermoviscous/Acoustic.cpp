#include "Acoustic.h"
#include <algorithm>
#include <numbers>
#include <numeric>
#include <stdexcept>

extern "C" {
void hfmm3d_t_c_p_(double *, std::complex<double> *, int64_t *, const double *, const std::complex<double> *, int64_t *, const double *, std::complex<double> *, int64_t *);
void hfmm3d_t_d_p_(double *, std::complex<double> *, int64_t *, const double *, const std::complex<double> *, int64_t *, const double *, std::complex<double> *, int64_t *);
void lfmm3d_t_d_p_(double *, int64_t *, const double *, const double *, int64_t *, const double *, double *, int64_t *);
}
namespace thermoviscous {
Vector Fmm(const FastData &d, Complex k, const Vector &coefficients, bool doubleLayer, double tolerance) {
    int64_t ns = d.Sources.cols(), nt = d.Targets.cols(), error = 0;
    Vector result(nt);
    if (k == Complex(0)) {
        Points dipoles = d.Normals;
        for (int j = 0; j < ns; ++j) dipoles.col(j) *= coefficients[j].real();
        Eigen::VectorXd real(nt);
        lfmm3d_t_d_p_(&tolerance, &ns, d.Sources.data(), dipoles.data(), &nt, d.Targets.data(), real.data(), &error);
        result = real.cast<Complex>();
    } else if (doubleLayer) {
        Eigen::Matrix<Complex, 3, Eigen::Dynamic> dipoles(3, ns);
        for (int j = 0; j < ns; ++j) dipoles.col(j) = d.Normals.col(j) * coefficients[j];
        hfmm3d_t_d_p_(&tolerance, &k, &ns, d.Sources.data(), dipoles.data(), &nt, d.Targets.data(), result.data(), &error);
    } else hfmm3d_t_c_p_(&tolerance, &k, &ns, d.Sources.data(), coefficients.data(), &nt, d.Targets.data(), result.data(), &error);
    if (error || !result.allFinite()) throw std::runtime_error("FMM3D evaluation failed: " + std::to_string(error));
    return result;
}
namespace {
struct Cluster {
    int Begin{}, End{}, Left{-1}, Right{-1};
    Point Low, High;
    int Size() const { return End - Begin; }
};
struct Tree {
    std::vector<int> Order;
    std::vector<Cluster> Nodes;
    Tree(const Points &points, int leaf) {
        Order.resize(points.cols());
        std::iota(Order.begin(), Order.end(), 0);
        const auto build = [&](auto &&self, int begin, int end) -> int {
            Cluster c{begin, end, -1, -1, Point::Constant(INFINITY), Point::Constant(-INFINITY)};
            for (int i = begin; i < end; ++i) {
                c.Low = c.Low.cwiseMin(points.col(Order[i]));
                c.High = c.High.cwiseMax(points.col(Order[i]));
            }
            const int node = int(Nodes.size());
            Nodes.push_back(c);
            if (end - begin > leaf) {
                Eigen::Index axis;
                (c.High - c.Low).maxCoeff(&axis);
                const int mid = (begin + end) / 2;
                std::nth_element(Order.begin() + begin, Order.begin() + mid, Order.begin() + end, [&](int a, int b) { return points(axis, a) < points(axis, b); });
                Nodes[node].Left = self(self, begin, mid);
                Nodes[node].Right = self(self, mid, end);
            }
            return node;
        };
        build(build, 0, int(Order.size()));
    }
};
struct Block {
    int Row{}, Col{};
    Matrix U, V;
};
class HMatrix {
    Tree Rows, Cols;
    std::vector<Block> Blocks;

public:
    HMatrix(const FastData &d, Complex k, bool doubleLayer, const Settings &s) : Rows(d.Targets, s.LeafSize), Cols(d.Sources, s.LeafSize) {
        const auto kernel = [&](int i, int j) -> Complex {
            const Point delta = d.Sources.col(Cols.Order[j]) - d.Targets.col(Rows.Order[i]);
            const double r = delta.norm();
            if (r == 0) return 0;
            const Complex ikr = Complex(0, 1) * k * r;
            const Complex g = std::exp(ikr) / (4 * std::numbers::pi * r);
            return doubleLayer ? g * (ikr - 1.) * delta.dot(d.Normals.col(Cols.Order[j])) / (r * r) : g;
        };
        const auto compress = [&](const Cluster &a, const Cluster &b) -> bool {
            const int m = a.Size(), n = b.Size(), limit = std::min(m, n);
            std::vector<Vector> us, vs;
            std::vector<bool> used(m), usedColumns(n);
            int pivotRow = 0;
            double norm2 = 0;
            while (int(us.size()) < limit) {
                const int rank = int(us.size());
                used[pivotRow] = true;
                Vector row(n);
                for (int j = 0; j < n; ++j) row[j] = kernel(a.Begin + pivotRow, b.Begin + j);
                for (int l = 0; l < rank; ++l) row -= us[l][pivotRow] * vs[l];
                int col = -1;
                double pivot = -1;
                for (int j = 0; j < n; ++j)
                    if (!usedColumns[j] && std::abs(row[j]) > pivot) {
                        pivot = std::abs(row[j]);
                        col = j;
                    }
                if (pivot <= 0) {
                    auto next = std::find(used.begin(), used.end(), false);
                    if (next == used.end()) break;
                    pivotRow = int(next - used.begin());
                    continue;
                }
                usedColumns[col] = true;
                row /= row[col];
                Vector column(m);
                for (int i = 0; i < m; ++i) column[i] = kernel(a.Begin + i, b.Begin + int(col));
                for (int l = 0; l < rank; ++l) column -= vs[l][col] * us[l];
                const double update = column.squaredNorm() * row.squaredNorm();
                for (int l = 0; l < rank; ++l) norm2 += 2 * std::real(us[l].dot(column) * vs[l].dot(row));
                norm2 += update;
                us.push_back(column);
                vs.push_back(row);
                double largest = -1;
                for (int i = 0; i < m; ++i)
                    if (!used[i] && std::abs(column[i]) > largest) {
                        largest = std::abs(column[i]);
                        pivotRow = i;
                    }
                if (rank >= 1 && update <= s.CompressionTolerance * s.CompressionTolerance * norm2) break;
                if ((rank + 1) * (m + n) >= m * n / 2) return false;
            }
            Block block{a.Begin, b.Begin, Matrix(m, us.size()), Matrix(vs.size(), n)};
            for (size_t i = 0; i < us.size(); ++i) {
                block.U.col(i) = us[i];
                block.V.row(i) = vs[i].transpose();
            }
            Blocks.push_back(std::move(block));
            return true;
        };
        const auto build = [&](auto &&self, int ai, int bi) -> void {
            const auto &a = Rows.Nodes[ai], &b = Cols.Nodes[bi];
            const double separation = (a.Low - b.High).cwiseMax(b.Low - a.High).cwiseMax(0).norm();
            const double diameter = std::max((a.High - a.Low).norm(), (b.High - b.Low).norm());
            if (diameter < 2 * separation && compress(a, b)) return;
            if (a.Left < 0 && b.Left < 0) {
                Block block{a.Begin, b.Begin, Matrix(a.Size(), b.Size()), {}};
                for (int j = 0; j < b.Size(); ++j)
                    for (int i = 0; i < a.Size(); ++i) block.U(i, j) = kernel(a.Begin + i, b.Begin + j);
                Blocks.push_back(std::move(block));
            } else if (b.Left < 0 || (a.Left >= 0 && a.Size() >= b.Size())) {
                self(self, a.Left, bi);
                self(self, a.Right, bi);
            } else {
                self(self, ai, b.Left);
                self(self, ai, b.Right);
            }
        };
        build(build, 0, 0);
    }
    Vector Apply(const Vector &x) const {
        Vector ordered(x.size()), result = Vector::Zero(Rows.Order.size());
        for (int j = 0; j < x.size(); ++j) ordered[j] = x[Cols.Order[j]];
        for (const auto &b : Blocks) {
            if (b.U.cols() == 0) continue;
            if (b.V.size()) result.segment(b.Row, b.U.rows()).noalias() += b.U * (b.V * ordered.segment(b.Col, b.V.cols()));
            else result.segment(b.Row, b.U.rows()).noalias() += b.U * ordered.segment(b.Col, b.U.cols());
        }
        Vector output(result.size());
        for (int i = 0; i < result.size(); ++i) output[Rows.Order[i]] = result[i];
        return output;
    }
    size_t Bytes() const {
        size_t bytes = 0;
        for (const auto &b : Blocks) bytes += (b.U.size() + b.V.size()) * sizeof(Complex);
        return bytes;
    }
};
class FastOperator final : public AcousticOperator {
    FastData Data;
    Complex K;
    Settings Options;
    std::unique_ptr<HMatrix> H, G;

public:
    FastOperator(FastData data, Complex k, const Settings &s) : Data(std::move(data)), K(k), Options(s) {
        if (s.AcousticBackend == "hmatrix") {
            H = std::make_unique<HMatrix>(Data, K, true, s);
            G = std::make_unique<HMatrix>(Data, K, false, s);
        }
    }
    Vector Apply(const Vector &x, bool doubleLayer) const override {
        const Vector coefficients = Data.Coefficients * x;
        Vector result = H ? (doubleLayer ? H->Apply(coefficients) : G->Apply(coefficients)) : Fmm(Data, K, coefficients, doubleLayer, Options.CompressionTolerance);
        if (doubleLayer) {
            result += Data.CorrectionH * x;
            if (!Options.Exterior) result = -result;
            result += Data.FreeTerm.cwiseProduct(x);
        } else result += Data.CorrectionG * x;
        return result;
    }
    size_t Bytes() const override {
        return (H ? H->Bytes() + G->Bytes() : 0) + (Data.Sources.size() + Data.Targets.size() + Data.Normals.size()) * sizeof(double) +
            (Data.Coefficients.nonZeros() + Data.CorrectionH.nonZeros() + Data.CorrectionG.nonZeros()) * (sizeof(Complex) + sizeof(int));
    }
};
} // namespace
std::unique_ptr<AcousticOperator> MakeAcoustic(FastData data, Complex k, const Settings &settings) { return std::make_unique<FastOperator>(std::move(data), k, settings); }
} // namespace thermoviscous
