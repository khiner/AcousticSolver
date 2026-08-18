#pragma once

// (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.
// RK4 integration of the coupled oscillator system M v''(t) + C v'(t) + K v(t) = F(t),
// over the all-pairs dense mass matrix with a Cholesky factorization of M.

#include "Oscillator.h"

#include <functional>

namespace FluidSound {

struct CoupledDirect {
    using Matrix = Eigen::MatrixXd;

    CoupledDirect(double dt) : Dt(dt) {}

    // One RK4 step. Stage inputs and the (k1 + 2 k2 + 2 k3 + k4) combination live in
    // preallocated scratch and accumulate in the reference's left-to-right element order,
    // so a step allocates nothing and stays bit-identical.
    void Step(double time);

    // Copies the oscillator data a batch needs, at its endpoint times. Call at batch start.
    void UpdateData(const std::vector<Oscillator *> &coupled, const std::vector<Oscillator *> &uncoupled, double time1, double time2);

    // Computes and factorizes M at the batch endpoints T1 and T2.
    void Refactor();

    // Solves v''(t) = M^-1 (F(t) - C v'(t) - K v(t)) for the packed state vectors [v v'].
    // `time` must satisfy T1 <= time <= T2. Returns Derivs, so a solve allocates nothing.
    const Eigen::ArrayXd &Solve(const Eigen::ArrayXd &states, double time);

    // Optional provider of precomputed endpoint mass-matrix INVERSES (see
    // BubbleFactorPipeline.h). Refactor calls it with (t1, t2, n_coupled); when it
    // succeeds, Solve applies the blended inverse as two matrix-vector products instead of
    // two pairs of triangular substitutions (same linear system, different float rounding).
    // When it returns false, Refactor computes Cholesky factors inline and Solve
    // substitutes, as in the reference.
    std::function<bool(double, double, int, Matrix &, Matrix &)> InverseProvider;

    // Mass matrix over packed solve data, shared by the inline path and the background
    // precompute workers. The row fill is parallel: each element is written exactly once,
    // with the reference's arithmetic.
    static void ConstructMass(const Eigen::Array<double, 6, Eigen::Dynamic> &solve_data1, const Eigen::Array<double, 6, Eigen::Dynamic> &solve_data2, double t1, double t2, double time, int n_coupled, Matrix &m);

    static constexpr double EpsSq{4.}; // Regularization term

    Eigen::ArrayXd States; // Packed state vectors [v ... v' ...] across all active oscillators
    Eigen::ArrayXd Derivs; // Packed derivatives [v' ... v'' ...] across all active oscillators

private:
    // Interpolates the stiffness, damping, and forcing terms at `time`.
    void ComputeKcf(double time);

    double Dt{0}; // Timestep size
    int NCoupled{0}; // Oscillators treated as coupled for the batch
    int NTotal{0}; // Total oscillators for the batch

    // Stiffness, damping, and forcing
    Eigen::ArrayXd KVals, CVals, FVals;

    double T1{-1}, T2{-1}; // Batch endpoint times

    // Oscillator::SolveData and ::ForceData, packed across active oscillators at the endpoints.
    Eigen::Array<double, 6, Eigen::Dynamic> SolveData1, SolveData2;
    Eigen::Array<double, 3, Eigen::Dynamic> ForceData1, ForceData2;

    Eigen::ArrayXd Y, Acc; // Preallocated RK4 scratch: stage input and running combination

    Eigen::ArrayXd Radii;
    Eigen::ArrayXd SqrtRadii; // sqrt(Radii), computed once per solve

    Eigen::LLT<Matrix> Factor1, Factor2;
    Eigen::VectorXd Rhs;

    // Precomputed endpoint inverses and their GEMV outputs, active for the current event
    // interval when UseInverse is set (see InverseProvider).
    Matrix Inv1, Inv2;
    Eigen::VectorXd Sol1, Sol2;
    bool UseInverse{false};
};

} // namespace FluidSound
