#pragma once

#include <complex>
#include <vector>

namespace immersed {

// A rational immittance in ascending powers of s. The coefficients are
// dimensionless and the denominator must not be the zero polynomial.
struct RationalImmittance {
    std::vector<double> Numerator;
    std::vector<double> Denominator;

    static RationalImmittance Constant(double value);
    static RationalImmittance Series(double resistance, double mass, double stiffness);
    RationalImmittance Reciprocal() const;
    std::complex<double> Evaluate(std::complex<double> s) const;
};

// Trapezoidal discretisation of the 2023 dual-forcing immittance. The transfer
// includes the factor two in p_D/Z0 = 2 z_v v_bar and v_D/Y0 = 2 y_p p_bar.
// Output() is split into Feedthrough()*input + History(), which is the form the
// immersed update needs before its new field sample is known.
class TrapezoidImmittance {
public:
    TrapezoidImmittance() = default;
    TrapezoidImmittance(const RationalImmittance &continuous, double time_step);

    double Feedthrough() const { return B.empty() ? 0. : B.front(); }
    double History() const;
    double PreviousOutput() const { return LastOutput; }
    void Step(double input);
    std::complex<double> Response(double angular_frequency) const;
    const std::vector<double> &Feedforward() const { return B; }
    const std::vector<double> &Feedback() const { return A; }

private:
    double TimeStep{0.};
    double LastOutput{0.};
    std::vector<double> B, A;
    std::vector<double> InputHistory, OutputHistory;
};

} // namespace immersed
