/** (c) 2024 Kangrui Xue
 *
 * \file Integrators.cpp
 * \brief Implements Integrator, Coupled_Direct, and Uncoupled classes
 * 
 * References:
 *   [Xue et al. 2023] Improved Water Sound Synthesis using Coupled Bubbles
 */

#include "Integrators.h"

#include "Parallel.h" // LOCAL PATCH (perf): bit-exact parallel fan-out helpers

// LOCAL PATCH (perf): AMX-backed matrix-vector products for the precomputed-inverse
// solve path (see InverseProvider in Integrators.h)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

namespace FluidSound {

/** LOCAL PATCH (perf, bit-exact): allocation-free RK4. Each stage's input is built in
 *  preallocated scratch and the combination (k1 + 2 k2 + 2 k3 + k4) accumulates in the
 *  original left-to-right element order, so results are bit-identical. */
template <typename T>
void Integrator<T>::step(double time)
{
    const Eigen::ArrayX<T>& k1 = solve(_States, time);
    _Acc = k1;
    _Y = _States + _dt / 2. * k1;
    const Eigen::ArrayX<T>& k2 = solve(_Y, time + _dt / 2.);
    _Acc += 2. * k2;
    _Y = _States + _dt / 2. * k2;
    const Eigen::ArrayX<T>& k3 = solve(_Y, time + _dt / 2.);
    _Acc += 2. * k3;
    _Y = _States + _dt * k3;
    const Eigen::ArrayX<T>& k4 = solve(_Y, time + _dt);
    _Acc += k4;

    _Derivs = _Acc / 6.;
    _States += _dt * _Derivs;
}

/** */
template <typename T>
void Integrator<T>::updateData(const std::vector<Oscillator<T>*>& coupled_osc, const std::vector<Oscillator<T>*>& uncoupled_osc,
    double time1, double time2)
{ 
    std::vector<Oscillator<T>*> total_osc(coupled_osc.begin(), coupled_osc.end());
    total_osc.insert(total_osc.end(), uncoupled_osc.begin(), uncoupled_osc.end());
    
    // Set high-level batch parameters: endpoints times and system size
    _N_coupled = coupled_osc.size(); _N_total = total_osc.size();
    _t1 = time1; _t2 = time2;

    // Set Oscillators solve data (radii, freq., xyz position, etc.) at endpoint times 
    _solveData1.resize(6, _N_total); _solveData2.resize(6, _N_total);
    for (int i = 0; i < _N_total; i++)
    {
        _solveData1.col(i) = total_osc[i]->interp(time1);
        _solveData2.col(i) = total_osc[i]->interp(time2);
    }

    // Set Oscillators force data at endpoint times
    _forceData1.resize(3, _N_coupled); _forceData2.resize(3, _N_coupled);
    for (int i = 0; i < _N_coupled; i++)
    {
        for (int forceIdx = 0; forceIdx < total_osc[i]->forceData.cols(); forceIdx++)
        {
            if (forceIdx == total_osc[i]->forceData.cols() - 1)
            {
                _forceData1.col(i) = total_osc[i]->forceData.col(forceIdx);
                _forceData2.col(i) = _forceData1.col(i);
            }
            else if (time1 < total_osc[i]->forceData(0, forceIdx + 1))
            {
                _forceData1.col(i) = total_osc[i]->forceData.col(forceIdx);
                _forceData2.col(i) = total_osc[i]->forceData.col(forceIdx + 1);
                break;
            }
        }
    } // FOR NOW, only coupled Oscillators are forced (uncoupled means Oscillator has ended, and we are waiting for it to die out)

    // Copy [v v'] from individual Oscillators into packed state vector representation
    _States.resize(2 * _N_total);
    for (int i = 0; i < _N_total; i++)
    {
        _States(i) = total_osc[i]->state(0);            // v - volume displacement
        _States(i + _N_total) = total_osc[i]->state(1); // v'- volume velocity
    }
    _Derivs.resize(2 * _N_total);
}

/** */
template <typename T>
void Integrator<T>::_computeKCF(double time)
{
    auto coeff_start = std::chrono::steady_clock::now();

    double alpha = (time - _t1) / (_t2 - _t1);

    // Compute stiffness at current time, K = w0^2
    // LOCAL PATCH (perf, bit-exact): fused .square() avoids the w0 temporary
    _Kvals = ((1. - alpha) * _solveData1.row(1) + alpha * _solveData2.row(1)).square();

    // Compute damping at current time (precomputed, so we just need to interpolate)
    _Cvals = (1. - alpha) * _solveData1.row(5) + alpha * _solveData2.row(5);

    // Compute force at current time (all forcing models we use have the form F(t) = (t < cutoff) * weight * t^2, \see Oscillators)
    _Fvals.setZero(_N_total); // LOCAL PATCH (perf, bit-exact): no realloc when the size is unchanged
    for (int i = 0; i < _N_coupled; i++)
    {
        if (time > _forceData2(0, i))
        {
            T cutoff = _forceData2(1, i); T weight = _forceData2(2, i); T t = time - _forceData2(0, i);
            _Fvals[i] = (t < cutoff) * weight * t * t;
        }
        else
        {
            T cutoff = _forceData1(1, i); T weight = _forceData1(2, i); T t = time - _forceData1(0, i);
            _Fvals[i] = (t < cutoff) * weight * t * t;
        }
    } // FOR NOW, assumes only coupled Oscillators are forced (uncoupled means Oscillator has ended, and we are waiting for it to die out)

    auto coeff_end = std::chrono::steady_clock::now();
    this->coeff_time += coeff_end - coeff_start;
}

template class Integrator<float>;
template class Integrator<double>;


/** LOCAL PATCH (perf, bit-exact): standalone construction shared by the inline path and
 *  the background precompute workers, with a parallel row fill (each element written
 *  exactly once, arithmetic unchanged). */
template <typename T>
void Coupled_Direct<T>::ConstructMass(const Eigen::Array<T, 6, Eigen::Dynamic>& solveData1, const Eigen::Array<T, 6, Eigen::Dynamic>& solveData2,
    double t1, double t2, double time, int N_coupled, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& M)
{
    double alpha = (time - t1) / (t2 - t1);

    // Precompute bubble centers and radii
    Eigen::Matrix<T, 3, Eigen::Dynamic, Eigen::RowMajor> centers(3, solveData1.cols());
    centers.row(0) = (1. - alpha) * solveData1.row(2) + alpha * solveData2.row(2);  // x
    centers.row(1) = (1. - alpha) * solveData1.row(3) + alpha * solveData2.row(3);  // y
    centers.row(2) = (1. - alpha) * solveData1.row(4) + alpha * solveData2.row(4);  // z

    Eigen::ArrayX<T> radii = (1. - alpha) * solveData1.row(0) + alpha * solveData2.row(0);

    // Dense, symmetric mass matrix M
    M.resize(N_coupled, N_coupled);
    ParallelChunks(N_coupled, 16, [&](size_t begin, size_t end)
    {
        for (int i = int(begin); i < int(end); ++i)
        {
            T r_i = radii[i];
            for (int j = i; j < N_coupled; ++j)
            {
                if (i == j) { M(i, j) = 1.; } // diagonal entries
                else {
                    T r_j = radii[j];
                    T distSq = (centers.col(j) - centers.col(i)).squaredNorm();
                    M(i, j) = 1. / std::sqrt(distSq / (r_i * r_j) + EpsSq);
                    M(j, i) = M(i, j);
                }
            }
        }
    });
}

template <typename T>
void Coupled_Direct<T>::_constructMass(double time, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M) const
{
    ConstructMass(_solveData1, _solveData2, _t1, _t2, time, _N_coupled, M);
}

/** LOCAL PATCH (perf): takes endpoint inverses from the precompute pipeline when one is
 *  installed, otherwise constructs and factors the two independent endpoints
 *  concurrently (the reference path). */
template <typename T>
void Coupled_Direct<T>::refactor()
{
    auto mass_start = std::chrono::steady_clock::now();

    if (InverseProvider && InverseProvider(_t1, _t2, _N_coupled, _inv1, _inv2))
    {
        _useInverse = true;
        auto mass_end = std::chrono::steady_clock::now();
        this->mass_time += mass_end - mass_start;
        return;
    }
    _useInverse = false;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> M1, M2;
    ParallelFor(2, 1, [&](size_t which)
    {
        if (which == 0) { _constructMass(_t1, M1); _factor1.compute(M1); }
        else { _constructMass(_t2, M2); _factor2.compute(M2); }
    });
    if (_factor1.info() == Eigen::NumericalIssue) { throw std::runtime_error("Non positive definite matrix!"); }
    if (_factor2.info() == Eigen::NumericalIssue) { throw std::runtime_error("Non positive definite matrix!"); }

    auto mass_end = std::chrono::steady_clock::now();
    this->mass_time += mass_end - mass_start;
}

/** We re-use _Derivs to avoid allocating additional memory */
template <typename T>
const Eigen::ArrayX<T>& Coupled_Direct<T>::solve(const Eigen::ArrayX<T>& States, double time)
{
    this->_computeKCF(time);

    auto solve_start = std::chrono::steady_clock::now();

    double alpha = (time - _t1) / (_t2 - _t1);
    _radii = (1. - alpha) * _solveData1.row(0) + alpha * _solveData2.row(0);
    _sqrtRadii = _radii.sqrt(); // LOCAL PATCH (perf, bit-exact): computed once, used twice

    // solve for y'' | My'' = (F/sqrt(r) - Cy' - Ky) (linearly interpolate M^{-1})
    _RHS = (_Fvals - _Cvals * States.segment(_N_total, _N_total) - _Kvals * States.segment(0, _N_total)) / _sqrtRadii;
    // LOCAL PATCH (perf): with precomputed endpoint inverses, the blended solve is two
    // matrix-vector products. Otherwise, substitute against the inline Cholesky factors.
    if (_useInverse) {
        if (_N_coupled > 0) {
            _sol1.resize(_N_coupled); _sol2.resize(_N_coupled);
            if constexpr (std::is_same_v<T, double>) {
                cblas_dgemv(CblasColMajor, CblasNoTrans, _N_coupled, _N_coupled, 1., _inv1.data(), _N_coupled, _RHS.data(), 1, 0., _sol1.data(), 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, _N_coupled, _N_coupled, 1., _inv2.data(), _N_coupled, _RHS.data(), 1, 0., _sol2.data(), 1);
            } else {
                cblas_sgemv(CblasColMajor, CblasNoTrans, _N_coupled, _N_coupled, 1.f, _inv1.data(), _N_coupled, _RHS.data(), 1, 0.f, _sol1.data(), 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans, _N_coupled, _N_coupled, 1.f, _inv2.data(), _N_coupled, _RHS.data(), 1, 0.f, _sol2.data(), 1);
            }
            _RHS.head(_N_coupled) = (1. - alpha) * _sol1 + alpha * _sol2;
        }
    } else {
        _RHS.head(_N_coupled) = (1. - alpha) * _factor1.solve(_RHS.head(_N_coupled)) + alpha * _factor2.solve(_RHS.head(_N_coupled));
    }

    _Derivs.segment(0, _N_total) = States.segment(_N_total, _N_total);
    _Derivs.segment(_N_total, _N_total) = _RHS.array() * _sqrtRadii;

    auto solve_end = std::chrono::steady_clock::now();
    this->solve_time += solve_end - solve_start;

    return _Derivs;
}

template class Coupled_Direct<float>;
template class Coupled_Direct<double>;


/** */
template <typename T>
const Eigen::ArrayX<T>& Uncoupled<T>::solve(const Eigen::ArrayX<T>& States, double time)
{
    this->_computeKCF(time);

    auto solve_start = std::chrono::steady_clock::now();

    _Derivs.segment(0, _N_total) = States.segment(_N_total, _N_total);
    _Derivs.segment(_N_total, _N_total) = _Fvals - _Cvals * States.segment(_N_total, _N_total) - _Kvals * States.segment(0, _N_total);

    auto solve_end = std::chrono::steady_clock::now();
    this->solve_time += solve_end - solve_start;

    return _Derivs;
}

template class Uncoupled<float>;
template class Uncoupled<double>;

} // namespace FluidSound
