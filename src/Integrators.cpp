// (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.
//
// References:
//   [Xue et al. 2023] Improved Water Sound Synthesis using Coupled Bubbles

#include "Integrators.h"

#include "Parallel.h"

// AMX-backed matrix-vector products for the precomputed-inverse solve path
// (see InverseProvider in Integrators.h)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <stdexcept>

namespace FluidSound {

void CoupledDirect::Step(double time) {
    const Eigen::ArrayXd &k1 = Solve(States, time);
    Acc = k1;
    Y = States + Dt / 2. * k1;
    const Eigen::ArrayXd &k2 = Solve(Y, time + Dt / 2.);
    Acc += 2. * k2;
    Y = States + Dt / 2. * k2;
    const Eigen::ArrayXd &k3 = Solve(Y, time + Dt / 2.);
    Acc += 2. * k3;
    Y = States + Dt * k3;
    const Eigen::ArrayXd &k4 = Solve(Y, time + Dt);
    Acc += k4;

    Derivs = Acc / 6.;
    States += Dt * Derivs;
}

void CoupledDirect::UpdateData(const std::vector<Oscillator *> &coupled, const std::vector<Oscillator *> &uncoupled, double time1, double time2) {
    std::vector<Oscillator *> total(coupled.begin(), coupled.end());
    total.insert(total.end(), uncoupled.begin(), uncoupled.end());

    NCoupled = coupled.size();
    NTotal = total.size();
    T1 = time1;
    T2 = time2;

    // Radii, frequency, and xyz position at the endpoint times
    SolveData1.resize(6, NTotal);
    SolveData2.resize(6, NTotal);
    for (int i = 0; i < NTotal; ++i) {
        SolveData1.col(i) = total[i]->Interp(time1);
        SolveData2.col(i) = total[i]->Interp(time2);
    }

    // Oscillator force data at the endpoint times. For now only coupled oscillators are
    // forced (uncoupled means the oscillator has ended and we are waiting for it to die out).
    ForceData1.resize(3, NCoupled);
    ForceData2.resize(3, NCoupled);
    for (int i = 0; i < NCoupled; ++i) {
        for (int f = 0; f < total[i]->ForceData.cols(); ++f) {
            if (f == total[i]->ForceData.cols() - 1) {
                ForceData1.col(i) = total[i]->ForceData.col(f);
                ForceData2.col(i) = ForceData1.col(i);
            } else if (time1 < total[i]->ForceData(0, f + 1)) {
                ForceData1.col(i) = total[i]->ForceData.col(f);
                ForceData2.col(i) = total[i]->ForceData.col(f + 1);
                break;
            }
        }
    }

    // Copy [v v'] from the individual oscillators into the packed state vector
    States.resize(2 * NTotal);
    for (int i = 0; i < NTotal; ++i) {
        States(i) = total[i]->State(0); // v, volume displacement
        States(i + NTotal) = total[i]->State(1); // v', volume velocity
    }
    Derivs.resize(2 * NTotal);
}

void CoupledDirect::ComputeKcf(double time) {
    const auto coeff_start = std::chrono::steady_clock::now();

    const double alpha = (time - T1) / (T2 - T1);

    // Stiffness at the current time, K = w0^2. The fused .square() avoids a w0 temporary.
    KVals = ((1. - alpha) * SolveData1.row(1) + alpha * SolveData2.row(1)).square();

    // Damping at the current time (precomputed, so this just interpolates)
    CVals = (1. - alpha) * SolveData1.row(5) + alpha * SolveData2.row(5);

    // Force at the current time. Every forcing model has the form
    // F(t) = (t < cutoff) * weight * t^2 (see Oscillator), and, as above, only coupled
    // oscillators are forced. setZero does not reallocate when the size is unchanged.
    FVals.setZero(NTotal);
    for (int i = 0; i < NCoupled; ++i) {
        const auto &force = time > ForceData2(0, i) ? ForceData2 : ForceData1;
        const double cutoff = force(1, i), weight = force(2, i), t = time - force(0, i);
        FVals[i] = (t < cutoff) * weight * t * t;
    }

    CoeffTime += std::chrono::steady_clock::now() - coeff_start;
}

void CoupledDirect::ConstructMass(const Eigen::Array<double, 6, Eigen::Dynamic> &solve_data1, const Eigen::Array<double, 6, Eigen::Dynamic> &solve_data2, double t1, double t2, double time, int n_coupled, Matrix &m) {
    const double alpha = (time - t1) / (t2 - t1);

    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> centers(3, solve_data1.cols());
    centers.row(0) = (1. - alpha) * solve_data1.row(2) + alpha * solve_data2.row(2); // x
    centers.row(1) = (1. - alpha) * solve_data1.row(3) + alpha * solve_data2.row(3); // y
    centers.row(2) = (1. - alpha) * solve_data1.row(4) + alpha * solve_data2.row(4); // z

    const Eigen::ArrayXd radii = (1. - alpha) * solve_data1.row(0) + alpha * solve_data2.row(0);

    m.resize(n_coupled, n_coupled);
    ParallelChunks(n_coupled, 16, [&](size_t begin, size_t end) {
        for (int i = int(begin); i < int(end); ++i) {
            const double r_i = radii[i];
            for (int j = i; j < n_coupled; ++j) {
                if (i == j) {
                    m(i, j) = 1.;
                } else {
                    const double dist_sq = (centers.col(j) - centers.col(i)).squaredNorm();
                    m(i, j) = 1. / std::sqrt(dist_sq / (r_i * radii[j]) + EpsSq);
                    m(j, i) = m(i, j);
                }
            }
        }
    });
}

// Takes endpoint inverses from the precompute pipeline when one is installed, otherwise
// constructs and factors the two independent endpoints concurrently (the reference path).
void CoupledDirect::Refactor() {
    const auto mass_start = std::chrono::steady_clock::now();

    if (InverseProvider && InverseProvider(T1, T2, NCoupled, Inv1, Inv2)) {
        UseInverse = true;
        MassTime += std::chrono::steady_clock::now() - mass_start;
        return;
    }
    UseInverse = false;
    Matrix m1, m2;
    ParallelFor(2, 1, [&](size_t which) {
        if (which == 0) {
            ConstructMass(SolveData1, SolveData2, T1, T2, T1, NCoupled, m1);
            Factor1.compute(m1);
        } else {
            ConstructMass(SolveData1, SolveData2, T1, T2, T2, NCoupled, m2);
            Factor2.compute(m2);
        }
    });
    if (Factor1.info() == Eigen::NumericalIssue || Factor2.info() == Eigen::NumericalIssue) throw std::runtime_error("Non positive definite matrix!");

    MassTime += std::chrono::steady_clock::now() - mass_start;
}

const Eigen::ArrayXd &CoupledDirect::Solve(const Eigen::ArrayXd &states, double time) {
    ComputeKcf(time);

    const auto solve_start = std::chrono::steady_clock::now();

    const double alpha = (time - T1) / (T2 - T1);
    Radii = (1. - alpha) * SolveData1.row(0) + alpha * SolveData2.row(0);
    SqrtRadii = Radii.sqrt();

    // Solve for y'': M y'' = (F/sqrt(r) - C y' - K y), linearly interpolating M^-1
    Rhs = (FVals - CVals * states.segment(NTotal, NTotal) - KVals * states.segment(0, NTotal)) / SqrtRadii;
    // With precomputed endpoint inverses the blended solve is two matrix-vector products.
    // Otherwise, substitute against the inline Cholesky factors.
    if (UseInverse) {
        if (NCoupled > 0) {
            Sol1.resize(NCoupled);
            Sol2.resize(NCoupled);
            cblas_dgemv(CblasColMajor, CblasNoTrans, NCoupled, NCoupled, 1., Inv1.data(), NCoupled, Rhs.data(), 1, 0., Sol1.data(), 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, NCoupled, NCoupled, 1., Inv2.data(), NCoupled, Rhs.data(), 1, 0., Sol2.data(), 1);
            Rhs.head(NCoupled) = (1. - alpha) * Sol1 + alpha * Sol2;
        }
    } else {
        Rhs.head(NCoupled) = (1. - alpha) * Factor1.solve(Rhs.head(NCoupled)) + alpha * Factor2.solve(Rhs.head(NCoupled));
    }

    Derivs.segment(0, NTotal) = states.segment(NTotal, NTotal);
    Derivs.segment(NTotal, NTotal) = Rhs.array() * SqrtRadii;

    SolveTime += std::chrono::steady_clock::now() - solve_start;

    return Derivs;
}

} // namespace FluidSound
