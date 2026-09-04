#include "ImmersedImmittance.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace immersed {
namespace {

void Trim(std::vector<double> &p) {
    while (p.size() > 1 && p.back() == 0.) p.pop_back();
}

bool IsZero(const std::vector<double> &p) {
    return std::ranges::all_of(p, [](double x) { return x == 0.; });
}

double Binomial(int n, int k) {
    if (k < 0 || k > n) return 0.;
    double value = 1.;
    for (int i = 1; i <= k; ++i) value *= double(n - k + i) / i;
    return value;
}

// p(s), after s = k(1-q)/(1+q), multiplied by (1+q)^order.
std::vector<double> BilinearPolynomial(const std::vector<double> &p, int order, double k) {
    std::vector<double> out(static_cast<size_t>(order) + 1, 0.);
    for (int power = 0; power < int(p.size()); ++power) {
        const double scale = p[size_t(power)] * std::pow(k, power);
        for (int minus_power = 0; minus_power <= power; ++minus_power) {
            const double minus = minus_power % 2 ? -1. : 1.;
            for (int plus_power = 0; plus_power <= order - power; ++plus_power) {
                out[size_t(minus_power) + size_t(plus_power)] +=
                    scale * minus * Binomial(power, minus_power) * Binomial(order - power, plus_power);
            }
        }
    }
    return out;
}

std::complex<double> EvaluatePolynomial(const std::vector<double> &p, std::complex<double> x) {
    std::complex<double> value{};
    for (auto it = p.rbegin(); it != p.rend(); ++it) value = value * x + *it;
    return value;
}

} // namespace

RationalImmittance RationalImmittance::Constant(double value) { return {{value}, {1.}}; }

RationalImmittance RationalImmittance::Series(double resistance, double mass, double stiffness) {
    if (resistance < 0. || mass < 0. || stiffness < 0.)
        throw std::runtime_error("A series immittance must have nonnegative resistance, mass and stiffness");
    // a + b s + c/s = (c + a s + b s^2) / s. The all-zero case is
    // represented directly so it has no artificial pole at DC.
    if (resistance == 0. && mass == 0. && stiffness == 0.) return Constant(0.);
    return {{stiffness, resistance, mass}, {0., 1.}};
}

RationalImmittance RationalImmittance::Reciprocal() const {
    if (IsZero(Numerator)) throw std::runtime_error("The zero immittance has no finite reciprocal");
    return {Denominator, Numerator};
}

std::complex<double> RationalImmittance::Evaluate(std::complex<double> s) const {
    const auto denominator = EvaluatePolynomial(Denominator, s);
    if (denominator == std::complex<double>{}) throw std::runtime_error("Immittance is singular at the requested frequency");
    return EvaluatePolynomial(Numerator, s) / denominator;
}

TrapezoidImmittance::TrapezoidImmittance(const RationalImmittance &continuous, double time_step)
    : TimeStep(time_step) {
    if (!(time_step > 0.)) throw std::runtime_error("Immittance time step must be positive");
    auto numerator = continuous.Numerator;
    auto denominator = continuous.Denominator;
    if (numerator.empty() || denominator.empty()) throw std::runtime_error("Immittance polynomials must not be empty");
    Trim(numerator);
    Trim(denominator);
    if (IsZero(denominator)) throw std::runtime_error("Immittance denominator is zero");

    const int order = int(std::max(numerator.size(), denominator.size()) - 1);
    B = BilinearPolynomial(numerator, order, 2. / time_step);
    A = BilinearPolynomial(denominator, order, 2. / time_step);
    const double a0 = A.front();
    if (!std::isfinite(a0) || a0 == 0.) throw std::runtime_error("Immittance bilinear transform has zero feedthrough denominator");
    for (double &b : B) b = 2. * b / a0;
    for (double &a : A) a /= a0;
    if (B.front() < -1e-12)
        throw std::runtime_error("A passive immittance cannot have negative trapezoidal feedthrough");
    B.front() = std::max(B.front(), 0.);
    InputHistory.assign(size_t(order), 0.);
    OutputHistory.assign(size_t(order), 0.);
}

double TrapezoidImmittance::History() const {
    double value = 0.;
    for (size_t i = 1; i < B.size(); ++i) value += B[i] * InputHistory[i - 1] - A[i] * OutputHistory[i - 1];
    return value;
}

void TrapezoidImmittance::Step(double input) {
    const double output = Feedthrough() * input + History();
    for (size_t i = InputHistory.size(); i > 1; --i) {
        InputHistory[i - 1] = InputHistory[i - 2];
        OutputHistory[i - 1] = OutputHistory[i - 2];
    }
    if (!InputHistory.empty()) {
        InputHistory.front() = input;
        OutputHistory.front() = output;
    }
    LastOutput = output;
}

std::complex<double> TrapezoidImmittance::Response(double angular_frequency) const {
    if (TimeStep == 0.) return {};
    const std::complex<double> q = std::exp(std::complex<double>{0., -angular_frequency * TimeStep});
    return EvaluatePolynomial(B, q) / EvaluatePolynomial(A, q);
}

} // namespace immersed
