#include "Plate.h"

#include <Eigen/Dense>
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace plate {
namespace {
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using View = Eigen::Map<const Matrix>;
constexpr double Pi = std::numbers::pi;

void Require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}

void CheckQ(const Prepared &p, std::span<const double> q) {
    Require(q.size() == p.Mx * p.My, "plate modal vector size mismatch");
    for (std::size_t i = 0; i < q.size(); ++i) {
        Require(std::isfinite(q[i]), "plate modal state must be finite");
        Require(p.Active[i] || q[i] == 0, "plate inactive modes must be zero");
    }
}

std::vector<double> Transform(std::size_t modes, std::size_t nodes, bool cosine) {
    const std::size_t rows = modes + (cosine ? 1 : 0);
    std::vector<double> result(rows * nodes);
    for (std::size_t i = 0; i < rows; ++i) {
        const double mode = double(i + (cosine ? 0 : 1));
        const double scale = cosine && i == 0 ? std::sqrt(1.0 / nodes) : std::sqrt(2.0 / nodes);
        for (std::size_t j = 0; j < nodes; ++j) {
            const double phase = Pi * mode * (double(j) + .5) / nodes;
            result[i * nodes + j] = scale * (cosine ? std::cos(phase) : std::sin(phase));
        }
    }
    return result;
}

Matrix Synthesize(const Prepared &p, const Matrix &q, bool cosine) {
    const auto &tx = cosine ? p.Cx : p.Sx;
    const auto &ty = cosine ? p.Cy : p.Sy;
    const std::size_t mx = p.Mx + (cosine ? 1 : 0), my = p.My + (cosine ? 1 : 0);
    // Separable transforms avoid a dense two-dimensional transform matrix.
    const Matrix temporary = View(tx.data(), mx, p.Nx).transpose() * q;
    return p.TransformScale * (temporary * View(ty.data(), my, p.Ny));
}

Matrix Analyze(const Prepared &p, const Matrix &grid, bool cosine) {
    const auto &tx = cosine ? p.Cx : p.Sx;
    const auto &ty = cosine ? p.Cy : p.Sy;
    const std::size_t mx = p.Mx + (cosine ? 1 : 0), my = p.My + (cosine ? 1 : 0);
    const Matrix temporary = View(tx.data(), mx, p.Nx) * grid;
    return (temporary * View(ty.data(), my, p.Ny).transpose()) / p.TransformScale;
}
} // namespace

Damping CalibrateDamping(const LossPoints &loss, double kappa) {
    Require(std::isfinite(kappa) && kappa > 0, "plate damping calibration requires positive finite kappa");
    for (const auto &point : loss)
        Require(std::isfinite(point[0]) && point[0] >= 0 && std::isfinite(point[1]) && point[1] > 0, "plate loss points require nonnegative finite frequencies and positive finite decay times");
    Require(loss[0][0] != loss[1][0], "plate loss calibration frequencies must differ");
    const double k0 = 2 * Pi * loss[0][0] / kappa, k1 = 2 * Pi * loss[1][0] / kappa;
    const double factor = 6 * std::numbers::ln10 / (k1 - k0);
    const Damping damping{factor * (k1 / loss[0][1] - k0 / loss[1][1]), factor * (-1 / loss[0][1] + 1 / loss[1][1])};
    Require(std::isfinite(damping.Sigma0) && damping.Sigma0 >= 0 && std::isfinite(damping.Sigma1) && damping.Sigma1 >= 0, "plate loss calibration must produce finite nonnegative damping coefficients");
    return damping;
}

PhysicalConversion FromPhysicalParameters(const PhysicalParameters &physical, Config settings) {
    for (const double value : {physical.Lx, physical.Ly, physical.Density, physical.Thickness, physical.YoungModulus})
        Require(std::isfinite(value) && value > 0, "plate dimensions, density and Young modulus must be positive and finite");
    Require(std::isfinite(physical.PoissonRatio) && physical.PoissonRatio > -1 && physical.PoissonRatio < .5, "plate Poisson ratio must lie strictly between -1 and 0.5");
    const double length = std::sqrt(physical.Lx * physical.Ly);
    const double poissonFactor = 1 - physical.PoissonRatio * physical.PoissonRatio;
    const double rigidity = physical.YoungModulus * std::pow(physical.Thickness, 3) / (12 * poissonFactor);
    settings.Kappa = std::sqrt(rigidity / (physical.Density * physical.Thickness)) / (length * length);
    settings.Ratio = physical.Lx / physical.Ly;
    const auto damping = CalibrateDamping(physical.Loss, settings.Kappa);
    settings.Sigma0 = damping.Sigma0;
    settings.Sigma1 = damping.Sigma1;
    const double output = physical.Thickness / std::sqrt(6 * poissonFactor);
    const double excitation = 1 / (physical.Density * physical.Thickness) / (length * length) / output;
    Require(std::isfinite(length) && length > 0 && std::isfinite(output) && output > 0 && std::isfinite(excitation) && excitation > 0 && std::isfinite(settings.Ratio) && settings.Ratio > 0, "plate physical scaling exceeds finite range");
    return {settings, length, excitation, output};
}

Prepared Prepare(const Config &c) {
    Require(std::isfinite(c.SampleRate) && c.SampleRate > 0, "plate sample rate must be positive and finite");
    const double cutoffLimit = c.UseExact ? c.SampleRate / 2 : c.SampleRate / Pi;
    Require(std::isfinite(c.CutoffHz) && c.CutoffHz >= 0 && c.CutoffHz <= cutoffLimit, "plate cutoff must be nonnegative and no greater than the time integrator's stability limit");
    const double cutoff = c.CutoffHz == 0 ? cutoffLimit : c.CutoffHz;
    Require(std::isfinite(c.Kappa) && c.Kappa > 0, "plate kappa must be positive and finite");
    Require(std::isfinite(c.Ratio) && c.Ratio > 0, "plate aspect ratio must be positive and finite");
    Require(std::isfinite(c.Sigma0) && c.Sigma0 >= 0 && std::isfinite(c.Sigma1) && c.Sigma1 >= 0, "plate damping must be nonnegative and finite");
    Require(std::isfinite(c.Lambda) && c.Lambda >= 0, "plate drift regulation must be nonnegative and finite");
    Require(std::isfinite(c.Epsilon) && c.Epsilon > 0, "plate regularization must be positive and finite");
    Require(std::isfinite(c.Oversampling) && c.Oversampling >= 1, "plate oversampling must be finite and at least 1");
    Require(c.NormType == 1 || c.NormType == 2, "plate drift regulator norm must be 1 or 2");
    // Preserve the reference cutoff rounding at integer mode boundaries.
    const double betaLimit = std::sqrt(2 * Pi * cutoff / c.Kappa);
    const double waveLimit = betaLimit * betaLimit;
    const double mx = std::floor(std::sqrt(std::max(0.0, (waveLimit / (Pi * Pi) - c.Ratio) * c.Ratio)));
    const double my = std::floor(std::sqrt(std::max(0.0, (waveLimit / (Pi * Pi) - 1 / c.Ratio) / c.Ratio)));
    Require(mx >= 1 && my >= 1, "plate cutoff excludes the fundamental mode");
    Require(mx <= 512 && my <= 512, "plate transform axes exceed supported maximum of 512 modes");
    const double nx = std::ceil(c.Oversampling * mx) + 1, ny = std::ceil(c.Oversampling * my) + 1;
    Require(nx <= 2048 && ny <= 2048, "plate collocation axes exceed supported maximum of 2048 nodes");
    Prepared p;
    p.Settings = c;
    p.Settings.CutoffHz = cutoff;
    p.Mx = std::size_t(mx);
    p.My = std::size_t(my);
    p.Nx = std::size_t(nx);
    p.Ny = std::size_t(ny);
    p.Dt = 1 / c.SampleRate;
    p.TransformScale = std::sqrt(double(p.Nx * p.Ny));
    const std::size_t count = p.Mx * p.My;
    p.Active.resize(count);
    p.AiryActive.resize((p.Mx + 1) * (p.My + 1));
    p.Kx.resize(count);
    p.Ky.resize(count);
    p.Omega2.resize(count);
    p.Sigma.resize(count);
    p.D.resize(count);
    for (std::size_t x = 0; x < p.Mx; ++x)
        for (std::size_t y = 0; y < p.My; ++y) {
            const auto i = x * p.My + y;
            const double kx = (x + 1) * Pi / std::sqrt(c.Ratio), ky = (y + 1) * Pi * std::sqrt(c.Ratio);
            const double k2 = kx * kx + ky * ky;
            p.Kx[i] = kx;
            p.Ky[i] = ky;
            p.Active[i] = std::sqrt(k2) < betaLimit;
            p.ActiveCount += p.Active[i];
            p.AiryActive[(x + 1) * (p.My + 1) + y + 1] = p.Active[i];
            const double omega = c.Kappa * k2, sigma = c.Sigma0 + c.Sigma1 * k2;
            Require(std::isfinite(omega) && std::isfinite(sigma), "plate modal coefficients overflow");
            if (c.UseExact) {
                const double decay = std::exp(-sigma * p.Dt);
                double numerator;
                if (omega >= sigma) {
                    const double phase = std::sqrt((omega - sigma) * (omega + sigma)) * p.Dt;
                    const double sine = std::sin(phase / 2);
                    numerator = (1 - decay) * (1 - decay) + 4 * decay * sine * sine;
                } else {
                    // Analytic continuation of cos to cosh, evaluated without overflow.
                    const double rate = std::sqrt((sigma - omega) * (sigma + omega));
                    const double slow = omega * (omega / (sigma + rate));
                    numerator = -std::expm1(-slow * p.Dt) * -std::expm1(-(sigma + rate) * p.Dt);
                }
                p.Omega2[i] = 2 * c.SampleRate * c.SampleRate * numerator / (1 + decay * decay);
                p.Sigma[i] = c.SampleRate * std::tanh(sigma * p.Dt);
            } else {
                p.Omega2[i] = omega * omega;
                p.Sigma[i] = sigma;
            }
            p.D[i] = 1 + p.Dt * p.Sigma[i];
            Require(std::isfinite(p.Omega2[i]) && std::isfinite((2 - p.Dt * p.Dt * p.Omega2[i]) / p.D[i]), "plate discretized coefficients overflow");
        }
    Require(p.ActiveCount > 0, "plate cutoff excludes all modes");
    for (std::size_t x = 1; x <= p.Mx; ++x) p.AiryActive[x * (p.My + 1)] = 1;
    for (std::size_t y = 1; y <= p.My; ++y) p.AiryActive[y] = 1;
    p.Sx = Transform(p.Mx, p.Nx, false);
    p.Sy = Transform(p.My, p.Ny, false);
    p.Cx = Transform(p.Mx, p.Nx, true);
    p.Cy = Transform(p.My, p.Ny, true);
    return p;
}

Backend SelectBackend(const Prepared &p) {
    if (!p.Settings.Nonlinear) return Backend::Cpu;
    const double x = p.Nx, y = p.Ny, m = p.Mx, n = p.My;
    const double denseWork = (3 * x * n * (m + y) + m * y * (x + n)) / 1e6;
    const auto fftWork = [](std::size_t nodes) {
        const bool direct = std::has_single_bit(nodes);
        const auto length = direct ? nodes : std::bit_ceil(2 * nodes - 1);
        return double(length) * std::countr_zero(length) * (direct ? 1 : 2);
    };
    const double fft = ((9 * n + 3 * y + 3) * fftWork(p.Nx) +
                        (9 * x + 3 * m + 2) * fftWork(p.Ny)) /
        1e6;
    const double modes = m * n / 1000;
    // Relative GPU transform-work estimate. Both choices use the same
    // high-precision arithmetic; FP64 CPU is only an explicit validation path
    // for nonlinear runs. See VALIDATION.md for the timing limitations.
    const double denseCost = 40 + 17 * denseWork + 15 * modes;
    const double fftCost = 75 + 20 * fft + 8 * modes;
    return denseCost <= fftCost ? Backend::MetalDense : Backend::MetalFft;
}

Potential EvaluatePotential(const Prepared &p, std::span<const double> q, bool gradient) {
    CheckQ(p, q);
    Matrix xx(p.Mx, p.My), yy(p.Mx, p.My), xy = Matrix::Zero(p.Mx + 1, p.My + 1);
    for (std::size_t x = 0; x < p.Mx; ++x)
        for (std::size_t y = 0; y < p.My; ++y) {
            const auto i = x * p.My + y;
            xx(x, y) = -p.Kx[i] * p.Kx[i] * q[i];
            yy(x, y) = -p.Ky[i] * p.Ky[i] * q[i];
            xy(x + 1, y + 1) = p.Kx[i] * p.Ky[i] * q[i];
        }
    const Matrix uxx = Synthesize(p, xx, false), uyy = Synthesize(p, yy, false), uxy = Synthesize(p, xy, true);
    Matrix airy = Analyze(p, (2 * (uxx.array() * uyy.array() - uxy.array().square())).matrix(), true);
    Matrix pxx(p.Mx + 1, p.My + 1), pyy(p.Mx + 1, p.My + 1), pxy(p.Mx, p.My);
    Potential result;
    for (std::size_t x = 0; x <= p.Mx; ++x)
        for (std::size_t y = 0; y <= p.My; ++y) {
            const double kx = x * Pi / std::sqrt(p.Settings.Ratio), ky = y * Pi * std::sqrt(p.Settings.Ratio);
            const double k2 = kx * kx + ky * ky;
            const double xi = p.AiryActive[x * (p.My + 1) + y] ? -airy(x, y) / (k2 * k2) : 0;
            airy(x, y) = xi;
            result.Value += .25 * (k2 * xi) * (k2 * xi);
            if (gradient) {
                pxx(x, y) = -kx * kx * xi;
                pyy(x, y) = -ky * ky * xi;
                if (x && y) pxy(x - 1, y - 1) = kx * ky * xi;
            }
        }
    result.Airy.assign(airy.data(), airy.data() + airy.size());
    if (!std::isfinite(result.Value)) throw std::runtime_error("plate nonlinear potential overflow");
    if (gradient) {
        const Matrix phixx = Synthesize(p, pxx, true), phiyy = Synthesize(p, pyy, true), phixy = Synthesize(p, pxy, false);
        const Matrix force = Analyze(p, (uxx.array() * phiyy.array() + uyy.array() * phixx.array() - 2 * uxy.array() * phixy.array()).matrix(), false);
        result.Gradient.resize(q.size());
        for (std::size_t i = 0; i < q.size(); ++i) {
            result.Gradient[i] = p.Active[i] ? -force.data()[i] : 0;
            if (!std::isfinite(result.Gradient[i])) throw std::runtime_error("plate nonlinear gradient overflow");
        }
    }
    return result;
}

State InitialState(const Prepared &p, std::span<const double> q, std::span<const double> previous) {
    CheckQ(p, q);
    CheckQ(p, previous);
    State state{{q.begin(), q.end()}, {previous.begin(), previous.end()}};
    std::vector<double> half(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) half[i] = .5 * (q[i] + previous[i]);
    state.Psi = std::sqrt(2 * EvaluatePotential(p, half, false).Value + p.Settings.Epsilon);
    return state;
}

State InitialState(const Prepared &p, double amplitude) {
    Require(std::isfinite(amplitude), "plate initial amplitude must be finite");
    std::vector<double> q(p.Mx * p.My);
    q[0] = amplitude;
    return InitialState(p, q, q);
}

void ValidatePoint(const Prepared &p, double x, double y) {
    Require(std::isfinite(x) && std::isfinite(y) && x >= 0 && y >= 0 && x <= std::sqrt(p.Settings.Ratio) && y <= 1 / std::sqrt(p.Settings.Ratio), "plate point must lie in the unit-area rectangular domain");
}

std::vector<double> PointBasis(const Prepared &p, double x, double y) {
    ValidatePoint(p, x, y);
    std::vector<double> basis(p.Mx * p.My);
    for (std::size_t i = 0; i < basis.size(); ++i)
        basis[i] = p.Active[i] ? 2 * std::sin(p.Kx[i] * x) * std::sin(p.Ky[i] * y) : 0;
    return basis;
}

double Pickup(const Prepared &p, const State &s, std::span<const double> basis) {
    CheckQ(p, s.Q);
    Require(basis.size() == s.Q.size(), "plate pickup basis size mismatch");
    double value = 0;
    for (std::size_t i = 0; i < basis.size(); ++i) value += basis[i] * s.Q[i];
    return value;
}

double RaisedCosineStrike(double time, double start, double duration, double amplitude, int type) {
    Require(std::isfinite(time) && std::isfinite(start) && std::isfinite(duration) && duration > 0 && std::isfinite(amplitude), "plate strike parameters must be finite with positive duration");
    Require(type >= 1, "plate cosine strike type must be a positive integer");
    const double local = time - start;
    return time < start || time > start + duration ? 0 : .5 * amplitude * (1 - std::cos(double(type) * Pi * local / duration));
}

double RepeatedCosineStrike(std::uint64_t sample, std::uint64_t cycleSamples, double sampleRate, double start, double duration, double amplitude, int type) {
    Require(cycleSamples > 0, "plate repeated excitation requires a nonempty sample cycle");
    Require(std::isfinite(sampleRate) && sampleRate > 0, "plate repeated excitation requires a positive finite sample rate");
    return RaisedCosineStrike(double(sample % cycleSamples) / sampleRate, start, duration, amplitude, type);
}

double Energy(const Prepared &p, const State &s) {
    CheckQ(p, s.Q);
    CheckQ(p, s.Previous);
    double energy = p.Settings.Nonlinear ? .5 * p.Settings.Kappa * p.Settings.Kappa * s.Psi * s.Psi : 0;
    for (std::size_t i = 0; i < s.Q.size(); ++i) {
        const double velocity = (s.Q[i] - s.Previous[i]) / p.Dt;
        energy += .5 * (velocity * velocity + p.Omega2[i] * s.Q[i] * s.Previous[i]);
    }
    if (!std::isfinite(energy)) throw std::runtime_error("plate numerical energy overflow");
    return energy;
}

Diagnostics Step(const Prepared &p, State &s, std::span<const double> force) {
    CheckQ(p, s.Q);
    CheckQ(p, s.Previous);
    Require(force.empty() || force.size() == s.Q.size(), "plate force size mismatch");
    Require(std::isfinite(s.Psi), "plate auxiliary state must be finite");
    for (const double f : force) Require(std::isfinite(f), "plate force must be finite");
    Diagnostics out;
    out.Energy = Energy(p, s);
    out.Psi = s.Psi;
    const std::size_t size = s.Q.size();
    std::vector<double> g(size), half(size), delta(size), next(size), residual(size);
    double norm = 0;
    for (std::size_t i = 0; i < size; ++i) {
        half[i] = .5 * (s.Q[i] + s.Previous[i]);
        delta[i] = s.Q[i] - s.Previous[i];
        const double velocity = delta[i] / p.Dt;
        norm += p.Settings.NormType == 1 ? std::abs(velocity) : velocity * velocity;
    }
    if (p.Settings.Nonlinear) {
        const auto potential = EvaluatePotential(p, s.Q);
        out.HalfPotential = EvaluatePotential(p, half, false).Value;
        out.Drift = s.Psi - std::sqrt(2 * out.HalfPotential + p.Settings.Epsilon);
        const double denominator = std::sqrt(2 * potential.Value + p.Settings.Epsilon);
        const double control = -p.Settings.Lambda * out.Drift / (norm + p.Settings.Epsilon);
        for (std::size_t i = 0; i < size; ++i) {
            const double direction = p.Settings.NormType == 1 ? double((delta[i] > 0) - (delta[i] < 0)) : delta[i] / p.Dt;
            g[i] = potential.Gradient[i] / denominator + control * direction;
        }
    }
    const double dt2 = p.Dt * p.Dt;
    const double nonlinear = dt2 * p.Settings.Kappa * p.Settings.Kappa;
    const double beta = .25 * nonlinear;
    double gd = 0, gh = 0, gr = 0;
    for (std::size_t i = 0; i < size; ++i) gd += g[i] * delta[i];
    // Sherman-Morrison inversion of the diagonal plus rank-one SAV update.
    // Increment form reduces cancellation for low-frequency modes in FP32.
    for (std::size_t i = 0; i < size; ++i) {
        if (!p.Active[i]) continue;
        const double rhs = (1 - p.Dt * p.Sigma[i]) * delta[i] - dt2 * p.Omega2[i] * s.Q[i] + dt2 * (force.empty() ? 0 : force[i]) - nonlinear * g[i] * s.Psi - beta * g[i] * gd;
        residual[i] = rhs / p.D[i];
        gr += g[i] * residual[i];
        gh += g[i] * g[i] / p.D[i];
    }
    const double correction = beta * gr / (1 + beta * gh);
    double psiIncrement = 0;
    for (std::size_t i = 0; i < size; ++i) {
        const double increment = residual[i] - g[i] / p.D[i] * correction;
        next[i] = s.Q[i] + increment;
        psiIncrement += .5 * g[i] * (increment + delta[i]);
        const double velocity = (increment + delta[i]) / (2 * p.Dt);
        out.InputPower += velocity * (force.empty() ? 0 : force[i]);
        out.DissipatedPower += 2 * p.Sigma[i] * velocity * velocity;
        if (!std::isfinite(next[i])) throw std::runtime_error("plate update produced a nonfinite displacement");
    }
    s.Psi += psiIncrement;
    if (!std::isfinite(s.Psi)) throw std::runtime_error("plate update produced a nonfinite auxiliary state");
    s.Previous.swap(s.Q);
    s.Q.swap(next);
    ++s.StepIndex;
    return out;
}

} // namespace plate
