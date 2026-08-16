#pragma once

// (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.
// Numerical integration of the coupled (or uncoupled) oscillator system
// M v''(t) + C v'(t) + K v(t) = F(t).

#include "Oscillator.h"

#include <chrono>
#include <functional>
#include <iostream>

namespace FluidSound {

// Base RK4 integrator. Deriving types specify how to compute and factorize M, and how to
// efficiently solve v''(t) = M^-1 (F(t) - C v'(t) - K v(t)).
template<typename T> struct Integrator {
    Integrator(double dt) : Dt(dt) {}
    virtual ~Integrator() = default;

    // One RK4 step. Stage inputs and the (k1 + 2 k2 + 2 k3 + k4) combination live in
    // preallocated scratch and accumulate in the reference's left-to-right element order,
    // so a step allocates nothing and stays bit-identical.
    void Step(double time);

    // Copies the oscillator data a batch needs, at its endpoint times. Call at batch start.
    void UpdateData(const std::vector<Oscillator<T> *> &coupled, const std::vector<Oscillator<T> *> &uncoupled, double time1, double time2);

    // Computes and factorizes M at the batch endpoints T1 and T2.
    virtual void Refactor() = 0;

    // Solves v''(t) = M^-1 (F(t) - C v'(t) - K v(t)) for the packed state vectors [v v'].
    // `time` must satisfy T1 <= time <= T2. Returns Derivs, so a solve allocates nothing.
    virtual const Eigen::ArrayX<T> &Solve(const Eigen::ArrayX<T> &states, double time) = 0;

    Eigen::ArrayX<T> States; // Packed state vectors [v ... v' ...] across all active oscillators
    Eigen::ArrayX<T> Derivs; // Packed derivatives [v' ... v'' ...] across all active oscillators

    std::chrono::duration<double> CoeffTime{}, MassTime{}, SolveTime{};

    void PrintTimings() const {
        std::cout << "K,C,F time: " << CoeffTime.count() << std::endl;
        std::cout << "M^-1 time:  " << MassTime.count() << std::endl;
        std::cout << "Solve time: " << SolveTime.count() << std::endl;
    }

protected:
    void ComputeKcf(double time);

    double Dt{0}; // Timestep size
    int NCoupled{0}; // Oscillators treated as coupled for the batch
    int NTotal{0}; // Total oscillators for the batch

    // Stiffness, damping, and forcing
    Eigen::ArrayX<T> KVals, CVals, FVals;

    double T1{-1}, T2{-1}; // Batch endpoint times

    // Oscillator::SolveData and ::ForceData, packed across active oscillators at the endpoints.
    Eigen::Array<T, 6, Eigen::Dynamic> SolveData1, SolveData2;
    Eigen::Array<T, 3, Eigen::Dynamic> ForceData1, ForceData2;

    Eigen::ArrayX<T> Y, Acc; // Preallocated RK4 scratch: stage input and running combination
};

// Integrator for the all-pairs, dense coupled oscillator system, with a Cholesky
// factorization of M.
template<typename T> struct CoupledDirect : Integrator<T> {
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    CoupledDirect(double dt) : Integrator<T>(dt) {}

    void Refactor() override;
    const Eigen::ArrayX<T> &Solve(const Eigen::ArrayX<T> &states, double time) override;

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
    static void ConstructMass(const Eigen::Array<T, 6, Eigen::Dynamic> &solve_data1, const Eigen::Array<T, 6, Eigen::Dynamic> &solve_data2, double t1, double t2, double time, int n_coupled, Matrix &m);

    static constexpr T EpsSq{4.}; // Regularization term

private:
    Eigen::ArrayX<T> Radii;
    Eigen::ArrayX<T> SqrtRadii; // sqrt(Radii), computed once per solve

    Eigen::LLT<Matrix> Factor1, Factor2;
    Eigen::Vector<T, Eigen::Dynamic> Rhs;

    // Precomputed endpoint inverses and their GEMV outputs, active for the current event
    // interval when UseInverse is set (see InverseProvider).
    Matrix Inv1, Inv2;
    Eigen::Vector<T, Eigen::Dynamic> Sol1, Sol2;
    bool UseInverse{false};

    // Needed for templated class inheritance to work
    using Integrator<T>::NCoupled;
    using Integrator<T>::NTotal;
    using Integrator<T>::Derivs;
    using Integrator<T>::KVals;
    using Integrator<T>::CVals;
    using Integrator<T>::FVals;

    using Integrator<T>::T1;
    using Integrator<T>::T2;
    using Integrator<T>::SolveData1;
    using Integrator<T>::SolveData2;
};

// Integrator for the uncoupled oscillator system (M = Identity).
template<typename T> struct Uncoupled : Integrator<T> {
    Uncoupled(double dt) : Integrator<T>(dt) {}

    void Refactor() override {}
    const Eigen::ArrayX<T> &Solve(const Eigen::ArrayX<T> &states, double time) override;

private:
    // Needed for templated class inheritance to work
    using Integrator<T>::NTotal;
    using Integrator<T>::Derivs;
    using Integrator<T>::KVals;
    using Integrator<T>::CVals;
    using Integrator<T>::FVals;
};

} // namespace FluidSound
