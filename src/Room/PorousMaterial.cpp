#include "PorousMaterial.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace Porous {
namespace {
constexpr double TwoPi = 2 * std::numbers::pi;
constexpr double Pressure = 101320;
constexpr double BulkInfinity = Pressure * 29.64 / 21.17;
using Response = std::function<Complex(Complex)>;

double Frequency(double low, double high, int i, int count) {
    return low * std::pow(high / low, double(i) / (count - 1));
}

// Fit F(s)=F(0)+sum w*s/(s+eta). Fixing the DC limit avoids spurious static
// inverse density. Rates are logarithmic variables, so poles stay strictly stable.
// K and R additionally require w>=0, a passive relaxation representation.
Rational Fit(const Response &response, double dc, double scale, double low, double high, double tolerance, bool positive_weights, double seed_rate, double *achieved = nullptr) {
    constexpr int Samples = 192;
    std::vector<Complex> frequencies(Samples), values(Samples);
    for (int i = 0; i < Samples; ++i) {
        frequencies[i] = Complex(0, TwoPi * Frequency(low, high, i, Samples));
        values[i] = response(frequencies[i]);
    }
    double best_error = std::numeric_limits<double>::infinity();
    for (int poles = 2; poles <= 8; ++poles) {
        for (int seed = 0; seed < 2; ++seed) {
            Eigen::VectorXd parameters(2 * poles);
            Eigen::MatrixXd basis(2 * Samples, poles);
            Eigen::VectorXd target(2 * Samples);
            for (int j = 0; j < poles; ++j)
                parameters[poles + j] = std::log(seed == 0 ? Frequency(seed_rate * 0.2, seed_rate * 5, j, poles) : TwoPi * Frequency(low, high, j, poles));
            for (int i = 0; i < Samples; ++i) {
                Complex const y = (values[i] - dc) / std::abs(values[i]);
                target[i] = y.real();
                target[Samples + i] = y.imag();
                for (int j = 0; j < poles; ++j) {
                    Complex const z = scale * frequencies[i] / (frequencies[i] + std::exp(parameters[poles + j])) / std::abs(values[i]);
                    basis(i, j) = z.real();
                    basis(Samples + i, j) = z.imag();
                }
            }
            parameters.head(poles) = basis.colPivHouseholderQr().solve(target);
            if (positive_weights)
                for (int j = 0; j < poles; ++j)
                    parameters[j] = std::log(std::max(parameters[j], 1e-3));
            auto const evaluate = [&](const Eigen::VectorXd &p, Eigen::MatrixXd *jacobian) {
                Eigen::VectorXd residual(2 * Samples);
                if (jacobian) jacobian->resize(2 * Samples, 2 * poles);
                for (int i = 0; i < Samples; ++i) {
                    Complex sum = dc;
                    for (int j = 0; j < poles; ++j) {
                        double const rate = std::exp(p[poles + j]);
                        double const weight = positive_weights ? std::exp(p[j]) : p[j];
                        Complex const z = scale * frequencies[i] / (frequencies[i] + rate);
                        sum += weight * z;
                        if (jacobian) {
                            Complex dw = z / std::abs(values[i]);
                            Complex const dr = -weight * dw * rate / (frequencies[i] + rate);
                            if (positive_weights) dw *= weight;
                            (*jacobian)(i, j) = dw.real();
                            (*jacobian)(Samples + i, j) = dw.imag();
                            (*jacobian)(i, poles + j) = dr.real();
                            (*jacobian)(Samples + i, poles + j) = dr.imag();
                        }
                    }
                    Complex const r = (sum - values[i]) / std::abs(values[i]);
                    residual[i] = r.real();
                    residual[Samples + i] = r.imag();
                }
                return residual;
            };
            double damping = 1e-3;
            for (int iteration = 0; iteration < 300; ++iteration) {
                Eigen::MatrixXd jacobian;
                Eigen::VectorXd const residual = evaluate(parameters, &jacobian);
                Eigen::MatrixXd normal = jacobian.transpose() * jacobian;
                normal.diagonal().array() += damping * normal.diagonal().array().max(1e-12);
                Eigen::VectorXd const step = normal.ldlt().solve(-jacobian.transpose() * residual);
                Eigen::VectorXd candidate = parameters + step;
                for (int j = 0; j < poles; ++j) {
                    candidate[poles + j] = std::clamp(candidate[poles + j], -20., 30.);
                    if (positive_weights) candidate[j] = std::clamp(candidate[j], -30., 30.);
                }
                double const cost = evaluate(candidate, nullptr).squaredNorm();
                if (std::isfinite(cost) && cost < residual.squaredNorm()) {
                    parameters = candidate;
                    damping = std::max(1e-12, damping * 0.3);
                    if (step.norm() < 1e-9) break;
                } else {
                    damping *= 10;
                    if (damping > 1e12) break;
                }
            }
            Rational result{.Constant = dc};
            bool valid = true;
            for (int j = 0; j < poles; ++j) {
                double const weight = (positive_weights ? std::exp(parameters[j]) : parameters[j]) * scale;
                double const rate = std::exp(parameters[poles + j]);
                valid &= std::isfinite(weight) && std::isfinite(rate) && (!positive_weights || weight >= 0);
                result.Constant += weight;
                result.Poles.push_back({rate, -weight * rate});
            }
            valid &= result.Constant > 0;
            double error = 0;
            // Check a denser grid than the optimization, plus a broad passivity
            // sample outside the fit band. This is not a proof of FD stability.
            for (int i = 0; i < 1025; ++i) {
                Complex const s(0, TwoPi * Frequency(low, high, i, 1025));
                error = std::max(error, std::abs(result.Evaluate(s) - response(s)) / std::abs(response(s)));
                Complex const outside = result.Evaluate(
                    Complex(0, TwoPi * Frequency(low * 1e-3, high * 1e3, i, 1025))
                );
                // The scalar wave equation requires Q(s)/s to be positive real:
                // on the imaginary axis this is Im(Q)>=0, not Re(Q)>=0.
                valid &= (!positive_weights || outside.real() >= -scale * 1e-10) && outside.imag() >= -scale * 1e-10;
            }
            best_error = std::min(best_error, error);
            if (valid && error <= tolerance) {
                if (achieved) *achieved = error;
                return result;
            }
        }
    }
    throw std::runtime_error("Porous material fit failed tolerance/passivity checks; best error " + std::to_string(best_error));
}
} // namespace

Complex Rational::Evaluate(Complex s) const {
    Complex value = Constant;
    for (const auto &p : Poles) value += p.Residue / (s + p.Rate);
    return value;
}

Fluid AllardChampoux(double flow_resistivity, Complex s) {
    if (!(flow_resistivity > 0) || !std::isfinite(flow_resistivity) || s == Complex{})
        throw std::invalid_argument("Allard-Champoux requires positive resistivity and nonzero frequency");
    double const w = TwoPi * flow_resistivity / 1.2;
    Complex const inverse_x = w / s;
    Complex const viscous = std::sqrt(0.0364 * inverse_x * inverse_x + 0.1144 * inverse_x);
    Complex const thermal = std::sqrt(2.82 * inverse_x * inverse_x + 24.9 * inverse_x);
    return {1.2 + viscous, Pressure * (29.64 + thermal) / (21.17 + thermal)};
}

Material AirMaterial(double speed, double density) {
    if (!(speed > 0) || !(density > 0) || !std::isfinite(speed * density))
        throw std::invalid_argument("Air speed and density must be finite and positive");
    Material result;
    result.BulkModulus.Constant = density * speed * speed;
    result.InverseDensity.Constant = 1 / density;
    result.WaveSpeedSquared.Constant = speed * speed;
    return result;
}

Material PrepareMaterial(double sigma, double low, double high, double tolerance) {
    if (!(sigma > 0) || !(low > 0) || !(high > low) || !(tolerance > 0) || !std::isfinite(sigma + low + high + tolerance))
        throw std::invalid_argument("Porous fit requires positive resistivity, band and tolerance");
    Material result{.FlowResistivity = sigma, .MinFrequency = low, .MaxFrequency = high};
    auto const fit = [&](const Response &response, double dc, double scale, bool positive) {
        double error = 0;
        Rational rational = Fit(response, dc, scale, low, high, tolerance, positive, sigma, &error);
        result.MaxRelativeError = std::max(result.MaxRelativeError, error);
        return rational;
    };
    result.BulkModulus = fit([&](Complex s) { return AllardChampoux(sigma, s).BulkModulus; }, Pressure, BulkInfinity, true);
    result.InverseDensity = fit([&](Complex s) { return 1. / AllardChampoux(sigma, s).Density; }, 0, 1 / 1.2, true);
    result.WaveSpeedSquared = fit([&](Complex s) {
        Fluid const fluid = AllardChampoux(sigma, s);
        return fluid.BulkModulus / fluid.Density;
    },
                                  0, BulkInfinity / 1.2, false);
    return result;
}

Rational PrepareLinkInverseDensity(const Material &a, const Material &b, double air_density, double tolerance) {
    if (!(air_density > 0) || !(tolerance > 0) || !std::isfinite(air_density + tolerance))
        throw std::invalid_argument("Porous link density and tolerance must be positive");
    if (a.FlowResistivity == 0 && b.FlowResistivity == 0)
        return {.Constant = 2 / (1 / a.InverseDensity.Constant + 1 / b.InverseDensity.Constant)};
    if (a.FlowResistivity == b.FlowResistivity && a.MinFrequency == b.MinFrequency && a.MaxFrequency == b.MaxFrequency)
        return a.InverseDensity;
    const auto &porous = a.FlowResistivity > 0 ? a : b;
    const bool both = a.FlowResistivity > 0 && b.FlowResistivity > 0;
    const double low = both ? std::max(a.MinFrequency, b.MinFrequency) : porous.MinFrequency;
    const double high = both ? std::min(a.MaxFrequency, b.MaxFrequency) : porous.MaxFrequency;
    if (!(low > 0 && high > low)) throw std::invalid_argument("Porous link materials have no common fit band");
    const auto density = [](const Material &m, Complex s) -> Complex {
        return m.FlowResistivity > 0 ? AllardChampoux(m.FlowResistivity, s).Density : Complex(1 / m.InverseDensity.Constant);
    };
    return Fit([&](Complex s) { return 2. / (density(a, s) + density(b, s)); }, 0, 1 / air_density, low, high, tolerance, true, porous.FlowResistivity);
}

Scattering LayerResponse(std::span<const Layer> layers, double frequency, double angle, bool rigid, double speed, double density) {
    if (!(frequency > 0) || !(speed > 0) || !(density > 0) || !std::isfinite(frequency + speed + density + angle) || std::abs(angle) >= std::numbers::pi / 2)
        throw std::invalid_argument("Invalid porous layer frequency, air properties or angle");
    double omega = TwoPi * frequency, kt = omega / speed * std::sin(angle);
    double const impedance = density * speed / std::cos(angle);
    Complex a = 1, b = 0, c = 0, d = 1;
    for (const Layer &layer : layers) {
        if (!(layer.Thickness >= 0) || !std::isfinite(layer.Thickness) || !(layer.FlowResistivity >= 0))
            throw std::invalid_argument("Invalid porous layer thickness or resistivity");
        Fluid const fluid = layer.FlowResistivity > 0 ? AllardChampoux(layer.FlowResistivity, Complex(0, omega)) : Fluid{density, density * speed * speed};
        Complex normal_k = std::sqrt(omega * omega * fluid.Density / fluid.BulkModulus - kt * kt);
        if (normal_k.real() < 0 || normal_k.imag() > 0) normal_k = -normal_k;
        Complex const z = omega * fluid.Density / normal_k;
        Complex const cosine = std::cos(normal_k * layer.Thickness);
        Complex const sine = Complex(0, 1) * std::sin(normal_k * layer.Thickness);
        Complex aa = a * cosine + b * sine / z, bb = a * z * sine + b * cosine;
        Complex cc = c * cosine + d * sine / z, dd = c * z * sine + d * cosine;
        a = aa;
        b = bb;
        c = cc;
        d = dd;
    }
    Complex const denominator = rigid ? a + impedance * c : a + b / impedance + impedance * c + d;
    Complex const reflection = rigid ? (a - impedance * c) / denominator : (a + b / impedance - impedance * c - d) / denominator;
    Complex const transmission = rigid ? Complex{} : 2. / denominator;
    return {reflection, transmission, 1 - std::norm(reflection) - std::norm(transmission)};
}
} // namespace Porous
