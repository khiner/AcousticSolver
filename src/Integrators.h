/** (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.
 *
 * \file Integrators.h
 * \brief Defines Integrator classes for numerically integrating coupled (or uncoupled) Oscillator systems
 */

#ifndef _FS_INTEGRATORS_H
#define _FS_INTEGRATORS_H

#include <chrono>
#include <functional> // LOCAL PATCH (perf): precomputed-factor provider hook

#include "Oscillator.h"

namespace FluidSound {

/**
 * \class Integrator
 * \brief Base RK4 integrator for the oscillator system \f$ M \ddot{v}(t) + C \dot{v}(t) + Kv(t) = F(t) \f$
 *
 * Classes extending Integrator are responsible for specifying how to compute + factorize M,
 *   as well as how to efficiently solve for \f$ \ddot{v}(t) = M^{-1} ( F(t) - C \dot{v}(t) - K v(t) ) \f$
 */
template<typename T>
class Integrator {
public:
    Integrator(double dt) : _dt(dt) {}

    // LOCAL PATCH (not upstream): Solver::~Solver() deletes derived integrators through this
    // base pointer, which is undefined behavior without a virtual destructor. Clang at -O2
    // compiles that delete to a trap. Upstream (github.com/kangruix/FluidSound) is archived.
    virtual ~Integrator() = default;

    /** \brief Takes an RK4 integration step */
    void step(double time);

    // LOCAL PATCH (perf, bit-exact): solve() returns a reference to _Derivs and step()
    // accumulates the RK4 combination incrementally (same per-element operation order as
    // the original k1..k4 expression), so stepping allocates nothing.

    /**
     * \brief Copies all Oscillator data needed for the integration batch; must be called at start of batch.
     * \param[in]  coupled_osc    Oscillators to treat as coupled
     * \param[in]  uncoupled_osc  Oscillators to treat as uncoupled
     * \param[in]  time1          integration batch start time
     * \param[in]  time2          integration batch end time
     */
    void updateData(const std::vector<Oscillator<T> *> &coupled_osc, const std::vector<Oscillator<T> *> &uncoupled_osc, double time1, double time2);

    /** \brief Computes and factorizes mass matrix at batch endpoints _t1 and _t2 */
    virtual void refactor() = 0;

    /**
     * \brief Solves for \f$ \ddot{v}(t) = M^{-1} ( F(t) - C \dot{v}(t) - K v(t) ) \f$
     * \param[in]  State  packed state vectors [v v']
     * \param[in]  time   solve time t (must satisfy _t1 <= t <= _t2)
     */
    virtual const Eigen::ArrayX<T> &solve(const Eigen::ArrayX<T> &State, double time) = 0;

    const Eigen::ArrayX<T> &States() { return _States; }
    const Eigen::ArrayX<T> &Derivs() { return _Derivs; }

protected:
    double _dt = 0.; //!< timestep size
    int _N_coupled = 0; //!< number of Oscillators to treat as coupled for the batch
    int _N_total = 0; //!< total number of Oscillators for the batch

    Eigen::ArrayX<T> _States; //!< packed state vectors [v ... v' ...] (across all active Oscillators)
    Eigen::ArrayX<T> _Derivs; //!< packed derivatives [v' ... v'' ...] (across all active Oscillators)

    // Stiffness, damping, and forcing
    Eigen::ArrayX<T> _Kvals, _Cvals, _Fvals;
    void _computeKCF(double time);

    /** \brief Batch endpoint times */
    double _t1 = -1., _t2 = -1.;

    /** \brief packed solve data (across all active Oscillators) at endpoint times, \see Oscillators.solveData */
    Eigen::Array<T, 6, Eigen::Dynamic> _solveData1, _solveData2;

    /** \brief packed force data (across all active Oscillators) at endpoint times, \see Oscillators.forceData */
    Eigen::Array<T, 3, Eigen::Dynamic> _forceData1, _forceData2;

    // LOCAL PATCH (perf, bit-exact): preallocated RK4 scratch (stage input and running
    // combination), see step().
    Eigen::ArrayX<T> _Y, _Acc;

public:
    std::chrono::duration<double> coeff_time = std::chrono::duration<double>::zero();
    std::chrono::duration<double> mass_time = std::chrono::duration<double>::zero();
    std::chrono::duration<double> solve_time = std::chrono::duration<double>::zero();
};

/**
 * \class Coupled_Direct
 * \brief Integrator for all-pairs, dense coupled oscillator system, with Cholesky factorization of M
 */
template<typename T>
class Coupled_Direct : public Integrator<T> {
public:
    Coupled_Direct(double dt) : Integrator<T>(dt) {}

    void refactor() override;
    const Eigen::ArrayX<T> &solve(const Eigen::ArrayX<T> &States, double time) override;

    // LOCAL PATCH (perf): optional provider of precomputed endpoint mass-matrix
    // INVERSES (see BubbleFactorPipeline.h). refactor() calls it with
    // (t1, t2, N_coupled); when it succeeds, solve() applies the blended inverse as two
    // matrix-vector products instead of two pairs of triangular substitutions (same
    // linear system, different float rounding). When it returns false, refactor()
    // computes Cholesky factors inline and solve() substitutes, as in the reference.
    std::function<bool(double, double, int, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &)> InverseProvider;

    /** LOCAL PATCH (perf): standalone mass-matrix construction over packed solve data,
     *  shared by the inline path and the background precompute workers. */
    static void ConstructMass(const Eigen::Array<T, 6, Eigen::Dynamic> &solveData1, const Eigen::Array<T, 6, Eigen::Dynamic> &solveData2, double t1, double t2, double time, int N_coupled, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M);

    // LOCAL PATCH: renamed public (from private _epsSq) so the precompute pipeline's
    // mass construction can share it
    static constexpr T EpsSq = 4.; //!< regularization term

private:
    /** \private constructs mass matrix M
     *  LOCAL PATCH (perf): out-parameter + const so the two endpoint constructions can
     *  run concurrently. */
    void _constructMass(double time, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M) const;

    Eigen::ArrayX<T> _radii;
    Eigen::ArrayX<T> _sqrtRadii; // LOCAL PATCH (perf, bit-exact): sqrt(_radii), computed once per solve

    Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _factor1, _factor2;
    Eigen::Vector<T, Eigen::Dynamic> _RHS;

    // LOCAL PATCH (perf): precomputed endpoint inverses and their GEMV outputs, active
    // for the current event interval when _useInverse is set (see InverseProvider).
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _inv1, _inv2;
    Eigen::Vector<T, Eigen::Dynamic> _sol1, _sol2;
    bool _useInverse = false;

    // Needed for templated class inheritance to work
    using Integrator<T>::_N_coupled;
    using Integrator<T>::_N_total;
    using Integrator<T>::_Derivs;
    using Integrator<T>::_Kvals;
    using Integrator<T>::_Cvals;
    using Integrator<T>::_Fvals;

    using Integrator<T>::_t1;
    using Integrator<T>::_t2;
    using Integrator<T>::_solveData1;
    using Integrator<T>::_solveData2;
};

/**
 * \class Uncoupled
 * \brief Integrator for uncoupled oscillator system (M = Identity)
 */
template<typename T>
class Uncoupled : public Integrator<T> {
public:
    Uncoupled(double dt) : Integrator<T>(dt) {}

    void refactor() override {} // dummy function call
    const Eigen::ArrayX<T> &solve(const Eigen::ArrayX<T> &State, double time) override;

private:
    // Needed for templated class inheritance to work
    using Integrator<T>::_N_total;
    using Integrator<T>::_Derivs;
    using Integrator<T>::_Kvals;
    using Integrator<T>::_Cvals;
    using Integrator<T>::_Fvals;
};

} // namespace FluidSound

#endif // _FS_INTEGRATORS_H
