#include "Thermoviscous.h"
#include <Eigen/LU>
#include <stdexcept>
namespace thermoviscous {
Vector DirectPressure(const Problem &p, const Operators &a) {
    const int n = int(a.Gh.rows());
    if (a.Acoustic) throw std::invalid_argument("Direct systems require acoustic_backend=dense");
    const int dimension = p.Options.Method == "direct10" ? 10 : p.Options.Method == "direct4" ? 4 :
                                                                                                1;
    const double estimated = 16. * n * n * (2. * dimension * dimension + 20);
    if (estimated > p.Options.DirectMemoryGiB * 1024 * 1024 * 1024) throw std::invalid_argument("Direct system exceeds direct_memory_gib; use the iterative solver or raise the explicit memory budget");
    const auto &c = a.Physics;
    const Matrix dc = a.Dc, dr = a.Dr, nd = a.Nd;
    Matrix system;
    Vector rhs;
    if (dimension == 10) {
        system = Matrix::Zero(10 * n, 10 * n);
        rhs = Vector::Zero(10 * n);
        rhs.tail(3 * n) = p.Velocity;
        system.block(0, 0, n, n) = a.Ha;
        system.block(0, n, n, n) = a.Ga;
        system.block(n, 2 * n, n, n) = a.Hh;
        system.block(n, 3 * n, n, n) = a.Gh;
        for (int axis = 0; axis < 3; ++axis) {
            system.block((2 + axis) * n, (4 + axis) * n, n, n) = a.Hv;
            system.block((2 + axis) * n, (7 + axis) * n, n, n) = a.Gv;
        }
        system.block(5 * n, 4 * n, n, 3 * n) = dr;
        system.block(5 * n, 7 * n, n, 3 * n) = nd.adjoint();
        system.block(6 * n, 0, n, n).diagonal().setConstant(c.TauA);
        system.block(6 * n, 2 * n, n, n).diagonal().setConstant(c.TauH);
        system.block(7 * n, 0, 3 * n, n) = c.PhiA * dc;
        system.block(7 * n, n, 3 * n, n) = c.PhiA * nd;
        system.block(7 * n, 2 * n, 3 * n, n) = c.PhiH * dc;
        system.block(7 * n, 3 * n, 3 * n, n) = c.PhiH * nd;
        system.block(7 * n, 4 * n, 3 * n, 3 * n).setIdentity();
    } else {
        const Matrix thermal = Matrix(a.Gh).partialPivLu().solve(Matrix(a.Hh));
        const Matrix viscous = Matrix(a.Gv).partialPivLu().solve(Matrix(a.Hv));
        Matrix r = dr;
        for (int axis = 0; axis < 3; ++axis) r.middleCols(axis * n, n) -= nd.middleRows(axis * n, n).adjoint() * viscous;
        const Complex muh = c.TauA * c.PhiH / c.TauH, mua = c.PhiA - muh;
        if (dimension == 4) {
            system = Matrix::Zero(4 * n, 4 * n);
            rhs = Vector::Zero(4 * n);
            rhs.tail(3 * n) = p.Velocity;
            system.block(0, n, n, 3 * n) = r;
            system.block(n, 0, 3 * n, n) = mua * dc + nd * (muh * thermal - c.PhiA * a.Ga.partialPivLu().solve(a.Ha));
            system.block(n, n, 3 * n, 3 * n).setIdentity();
        } else {
            const Eigen::PartialPivLU<Matrix> rn(r * nd);
            system = a.Ga * (mua * rn.solve(Matrix(r * dc)) + muh * thermal) - c.PhiA * a.Ha;
            rhs = a.Ga * rn.solve(Vector(r * p.Velocity));
        }
    }
    const Vector solution = system.partialPivLu().solve(rhs);
    const double residual = (system * solution - rhs).norm() / std::max(rhs.norm(), 1e-300);
    if (!solution.allFinite() || residual > 1e-6) throw std::runtime_error("Direct coupled system failed its residual check");
    return solution.head(n);
}
} // namespace thermoviscous
