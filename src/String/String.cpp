#include "String.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace stiffstring {
namespace {
using Vector = std::vector<double>;
bool Geometric(const Config &c) { return c.Equation == Model::GeometricA || c.Equation == Model::GeometricB; }
double Square(double x) { return x * x; }
void RequirePositiveFinite(std::initializer_list<double> values) {
    for (double const value : values)
        if (!(value > 0) || !std::isfinite(value))
            throw std::invalid_argument("String parameters produce a non-finite or non-positive physical coefficient or grid");
}
double Dot(std::span<const double> a, std::span<const double> b) {
    double sum{};
    for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}
Vector Difference(const Prepared &p, std::span<const double> q) {
    Vector result(p.Intervals);
    for (std::size_t i = 0; i < p.Intervals; ++i)
        result[i] = ((i < p.Points ? q[i] : 0) - (i ? q[i - 1] : 0)) / p.H;
    return result;
}
Vector Divergence(const Prepared &p, std::span<const double> q) {
    Vector result(p.Points);
    for (std::size_t i = 0; i < p.Points; ++i) result[i] = (q[i + 1] - q[i]) / p.H;
    return result;
}
Vector Laplacian(const Prepared &p, std::span<const double> q) {
    Vector result(p.Points);
    for (std::size_t i = 0; i < p.Points; ++i)
        result[i] = ((i ? q[i - 1] : 0) - 2 * q[i] + (i + 1 < p.Points ? q[i + 1] : 0)) / Square(p.H);
    return result;
}
Vector Midpoint(std::span<const double> a, std::span<const double> b) {
    Vector result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) result[i] = 0.5 * (a[i] + b[i]);
    return result;
}
double Psi(const Prepared &p, double potential) {
    const auto &c = p.Settings;
    const double shift = c.Equation == Model::KirchhoffCarrier && c.Split ? 0 : (Geometric(c) ? c.Shift : 2 * c.Shift);
    const double radicand = 2 * potential + shift;
    if (!(radicand >= 0) || !std::isfinite(radicand)) throw std::runtime_error("String potential is not finite and nonnegative");
    return std::sqrt(radicand);
}
Vector LinearUpdate(const Prepared &p, const State &s, bool full) {
    Vector result(s.Q.size());
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = p.Mass * (2 * s.Q[i] - s.Previous[i]);
    if (full)
        for (std::size_t component = 0; component < p.Components; ++component) {
            const auto q = std::span(s.Q).subspan(component * p.Points, p.Points);
            const auto previous = std::span(s.Previous).subspan(component * p.Points, p.Points);
            auto lap = Laplacian(p, q), lapPrevious = Laplacian(p, previous), fourth = Laplacian(p, lap);
            const double tension = component ? p.LongitudinalTension : p.Settings.Tension;
            for (std::size_t i = 0; i < p.Points; ++i) {
                auto &value = result[component * p.Points + i];
                value += Square(p.Dt) * (tension * lap[i] - (component ? 0 : p.Rigidity * fourth[i]));
                value += p.Mass * p.Sigma0 * p.Dt * previous[i];
                if (!component) value += 2 * p.Mass * p.Sigma1 * p.Dt * (lap[i] - lapPrevious[i]);
            }
        }
    return result;
}
} // namespace

Config Preset(std::string_view name) {
    Config c;
    if (name == "ge-b-split") return c;
    if (name == "ge-a-split" || name == "ge-a-unsplit") {
        c.Equation = Model::GeometricA;
        c.Split = name == "ge-a-split";
        c.SampleRate = (c.Split ? 256 : 512) * 44100.0;
        c.Shift = c.Split ? std::numeric_limits<double>::epsilon() : 1000;
        c.Spacing = c.Split ? Grid::Longitudinal : Grid::Transverse;
    } else if (name == "cubic-split" || name == "cubic-unsplit") {
        c.Equation = Model::Cubic;
        c.Split = name == "cubic-split";
        c.SampleRate = 1024 * 44100.0;
        c.Initial = InitialCondition::FirstMode;
        c.Spacing = Grid::Transverse;
    } else if (name == "kc-split" || name == "kc-unsplit") {
        c.Equation = Model::KirchhoffCarrier;
        c.Split = name == "kc-split";
        c.SampleRate = (c.Split ? 2 : 512) * 44100.0;
        c.Initial = c.Split ? InitialCondition::RaisedCosine : InitialCondition::FirstMode;
        c.Shift = c.Split ? std::numeric_limits<double>::epsilon() : 1000;
        c.Spacing = Grid::Transverse;
    } else throw std::invalid_argument("Unknown string preset: " + std::string(name));
    return c;
}

Prepared Prepare(const Config &c) {
    RequirePositiveFinite({c.SampleRate, c.Density, c.Tension, c.Radius, c.YoungModulus, c.Length, c.T60Low, c.T60High, c.RaisedCosineWidth});
    const double nonnegative[]{c.Duration, c.Shift, c.StabilityMargin, c.LossFrequencyLow};
    for (double const x : nonnegative)
        if (!(x >= 0) || !std::isfinite(x)) throw std::invalid_argument("String duration, shift, margin and loss frequency must be finite and nonnegative");
    if (!std::isfinite(c.Amplitude) || !std::isfinite(c.LossFrequencyHigh) || c.LossFrequencyHigh <= c.LossFrequencyLow)
        throw std::invalid_argument("Invalid string amplitude or damping frequencies");
    if (c.Equation == Model::GeometricB && !c.Split) throw std::invalid_argument("Geometric Form B requires split potential");
    if (!Geometric(c) && c.Spacing == Grid::Longitudinal) throw std::invalid_argument("Longitudinal grid applies only to geometric strings");
    if (Geometric(c) && !c.Nonlinear) throw std::invalid_argument("Use the cubic or Kirchhoff-Carrier model for the linear limit");
    Prepared p{.Settings = c};
    p.Area = std::numbers::pi * Square(c.Radius);
    p.Mass = c.Density * p.Area;
    p.EA = c.YoungModulus * p.Area;
    p.Rigidity = c.YoungModulus * std::numbers::pi * Square(Square(c.Radius)) / 4;
    p.NonlinearModulus = p.EA - c.Tension;
    if (!(p.NonlinearModulus > 0)) throw std::invalid_argument("String axial stiffness must exceed tension");
    p.LongitudinalTension = c.Equation == Model::GeometricB ? p.EA : c.Tension;
    const double c2 = c.Tension / p.Mass, stiffness2 = p.Rigidity / p.Mass;
    RequirePositiveFinite({p.Area, p.Mass, p.EA, p.Rigidity, p.NonlinearModulus, c2, stiffness2, c.YoungModulus / c.Density});
    if (c.Damped) {
        auto const waveNumber2 = [&](double frequency) {
            const double omega = 2 * std::numbers::pi * frequency;
            return (-c2 + std::sqrt(Square(c2) + 4 * stiffness2 * Square(omega))) / (2 * stiffness2);
        };
        const double a = waveNumber2(c.LossFrequencyLow), b = waveNumber2(c.LossFrequencyHigh);
        RequirePositiveFinite({b - a});
        p.Sigma0 = 6 * std::numbers::ln10 / (b - a) * (b / c.T60Low - a / c.T60High);
        p.Sigma1 = 6 * std::numbers::ln10 / (b - a) * (-1 / c.T60Low + 1 / c.T60High);
        if (!std::isfinite(p.Sigma0) || !std::isfinite(p.Sigma1) || p.Sigma0 < 0 || p.Sigma1 < 0)
            throw std::invalid_argument("String loss points imply non-finite or negative damping");
    }
    const double requestedDt = 1 / c.SampleRate, margin = 1 + c.StabilityMargin;
    const double term = c2 * Square(requestedDt) + 4 * p.Sigma1 * requestedDt;
    const double h = c.Spacing == Grid::Longitudinal ? std::sqrt(c.YoungModulus / c.Density) * requestedDt : std::sqrt((term + std::sqrt(Square(term) + 16 * stiffness2 * Square(requestedDt))) / 2) * margin;
    RequirePositiveFinite({requestedDt, margin, h});
    const std::size_t multiple = Geometric(c) ? 4 : 2;
    const double estimated = multiple * std::floor(c.Length / h / multiple);
    if (!std::isfinite(estimated) || estimated < 0 || estimated > 1000000) throw std::invalid_argument("String requested grid is invalid or exceeds one million intervals");
    p.Intervals = c.Intervals ? c.Intervals : static_cast<std::size_t>(estimated);
    if (p.Intervals < 4 || p.Intervals > 1000000 || p.Intervals % multiple)
        throw std::invalid_argument("String intervals must be a multiple of four (geometric) or two, at least four and at most one million");
    p.Points = p.Intervals - 1;
    p.Components = Geometric(c) ? 2 : 1;
    p.H = c.Length / p.Intervals;
    RequirePositiveFinite({p.H, Square(p.H)});
    const double width = std::floor(c.RaisedCosineWidth * p.Points);
    if (!std::isfinite(width) || (c.Initial == InitialCondition::RaisedCosine && width < 1))
        throw std::invalid_argument("Raised cosine width must contain at least one point and have finite extent");
    if (c.Spacing == Grid::Longitudinal) p.Dt = 0.999 * p.H / std::sqrt(c.YoungModulus / c.Density);
    else {
        // Algebraically the upstream inversion of the transverse stability bound.
        p.Dt = Square(p.H) * (std::sqrt(4 * stiffness2 * Square(margin) + c2 * Square(p.H) + 4 * Square(p.Sigma1 * margin)) - 2 * p.Sigma1 * margin) / (4 * stiffness2 * margin * margin * margin + c2 * margin * Square(p.H));
        if (Geometric(c)) p.Dt *= 0.999;
    }
    p.SampleRate = std::floor(1 / p.Dt);
    RequirePositiveFinite({p.Dt, Square(p.Dt), p.SampleRate});
    const double samples = std::floor(p.SampleRate * c.Duration);
    if (!std::isfinite(samples) || samples < 0 || samples >= static_cast<double>(std::numeric_limits<std::size_t>::max() - 2))
        throw std::invalid_argument("String duration exceeds addressable sample count");
    p.SampleCount = static_cast<std::size_t>(samples);
    p.Receiver = {p.Intervals / 2 - 1, (p.Intervals / 2 + 1) / 2 - 1};
    p.InverseMass = 1 / (p.Mass * (1 + p.Sigma0 * p.Dt));
    RequirePositiveFinite({p.InverseMass});
    return p;
}

Potential EvaluatePotential(const Prepared &p, std::span<const double> x, bool gradient) {
    if (x.size() != p.Points * p.Components) throw std::invalid_argument("String state size does not match grid");
    const auto &c = p.Settings;
    Potential result{.Gradient = gradient ? Vector(x.size()) : Vector{}};
    const auto u = x.first(p.Points);
    auto lap = !c.Split || c.Equation == Model::KirchhoffCarrier ? Laplacian(p, u) : Vector{};
    if (c.Equation == Model::KirchhoffCarrier) {
        const double norm = -Dot(u, lap);
        if (c.Nonlinear) {
            result.Value = p.EA * Square(p.H) / (8 * c.Length) * Square(norm);
            if (gradient)
                for (std::size_t i = 0; i < p.Points; ++i)
                    result.Gradient[i] = -0.5 * p.EA * Square(p.H) / c.Length * norm * lap[i];
        }
    } else if (Geometric(c) || c.Nonlinear) {
        auto q = Difference(p, u), slope = Geometric(c) ? Difference(p, x.subspan(p.Points)) : Vector{};
        Vector transverse(p.Intervals), longitudinal(Geometric(c) ? p.Intervals : 0);
        for (std::size_t i = 0; i < p.Intervals; ++i) {
            if (c.Equation == Model::Cubic) {
                result.Value += p.H * p.NonlinearModulus / 8 * Square(Square(q[i]));
                transverse[i] = p.H * p.NonlinearModulus / 2 * q[i] * q[i] * q[i];
            } else {
                const double stretch = std::sqrt(Square(1 + slope[i]) + Square(q[i]));
                if (!(stretch > 0)) throw std::runtime_error("Geometric string has zero stretch");
                if (c.Equation == Model::GeometricA) {
                    result.Value += 0.5 * p.H * p.NonlinearModulus * Square(stretch - 1);
                    transverse[i] = p.H * p.NonlinearModulus * (stretch - 1) * q[i] / stretch;
                    longitudinal[i] = p.H * p.NonlinearModulus * (stretch - 1) * (1 + slope[i]) / stretch;
                } else {
                    result.Value += p.H * p.NonlinearModulus * (0.5 * Square(q[i]) + slope[i] + 1.5 - stretch);
                    transverse[i] = p.H * p.NonlinearModulus * (q[i] - q[i] / stretch);
                    longitudinal[i] = p.H * p.NonlinearModulus * (1 - (1 + slope[i]) / stretch);
                }
            }
        }
        if (gradient)
            for (std::size_t component = 0; component < p.Components; ++component) {
                const auto flux = Divergence(p, component ? longitudinal : transverse);
                for (std::size_t i = 0; i < p.Points; ++i) result.Gradient[component * p.Points + i] = -flux[i];
            }
    }
    if (!c.Split) {
        for (std::size_t component = 0; component < p.Components; ++component) {
            const auto part = x.subspan(component * p.Points, p.Points);
            auto d2 = component ? Laplacian(p, part) : lap;
            const double tension = component ? p.LongitudinalTension : c.Tension;
            result.Value -= 0.5 * tension * p.H * Dot(part, d2);
            if (!component) result.Value += 0.5 * p.Rigidity * p.H * Dot(d2, d2);
            if (gradient) {
                auto d4 = Laplacian(p, d2);
                for (std::size_t i = 0; i < p.Points; ++i)
                    result.Gradient[component * p.Points + i] += -tension * p.H * d2[i] + (component ? 0 : p.Rigidity * p.H * d4[i]);
            }
        }
    }
    return result;
}

State InitialState(const Prepared &p) {
    const auto &c = p.Settings;
    State state{.Q = Vector(p.Points * p.Components), .Previous = Vector(p.Points * p.Components)};
    const double width = std::floor(c.RaisedCosineWidth * p.Points);
    for (std::size_t i = 0; i < p.Points; ++i) {
        const double offset = static_cast<double>(i) - p.Receiver[0];
        state.Previous[i] = c.Amplitude * (c.Initial == InitialCondition::FirstMode ? std::sin(std::numbers::pi * (i + 1) / p.Intervals) : (std::abs(offset) <= width ? 0.5 * (1 + std::cos(std::numbers::pi * offset / width)) : 0));
    }
    auto const u = std::span(state.Previous).first(p.Points);
    auto q = Difference(p, u), lap = Laplacian(p, u), fourth = Laplacian(p, lap);
    Vector flux(p.Intervals), longFlux(p.Intervals);
    for (std::size_t i = 0; i < p.Intervals; ++i) {
        const double stretch = std::sqrt(1 + Square(q[i]));
        if (c.Equation == Model::GeometricA) {
            flux[i] = p.H * p.NonlinearModulus * (stretch - 1) * q[i] / stretch;
            longFlux[i] = p.H * p.NonlinearModulus * (stretch - 1) / stretch;
        } else if (c.Equation == Model::GeometricB) {
            // Preserve the upstream startup, whose flux differs from later updates.
            flux[i] = 2 * p.H * p.NonlinearModulus * (stretch - 1) * (q[i] - q[i] / stretch);
            longFlux[i] = 2 * p.H * p.NonlinearModulus * (stretch - 1) * (1 - 1 / stretch);
        } else if (c.Equation == Model::Cubic && c.Nonlinear) flux[i] = p.H * p.NonlinearModulus * 0.5 * q[i] * q[i] * q[i];
    }
    auto force = Divergence(p, flux), longForce = Divergence(p, longFlux);
    const double norm = -Dot(u, lap);
    for (std::size_t i = 0; i < p.Points; ++i) {
        double nonlinear = Geometric(c) ? p.InverseMass * force[i] : force[i];
        if (c.Equation == Model::KirchhoffCarrier && c.Nonlinear)
            nonlinear = p.H * 0.5 * c.YoungModulus * p.H / (c.Length * c.Density) * norm * lap[i];
        state.Q[i] = state.Previous[i] + 0.5 * Square(p.Dt) * (p.H * c.Tension / p.Mass * lap[i] - p.H * p.Rigidity / p.Mass * fourth[i] + nonlinear);
        if (p.Components == 2) state.Q[p.Points + i] = 0.5 * Square(p.Dt) * p.InverseMass * longForce[i];
    }
    if (!std::ranges::all_of(state.Q, [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("String amplitude produces a non-finite initial state");
    state.Psi = Psi(p, EvaluatePotential(p, Midpoint(state.Q, state.Previous), false).Value);
    return state;
}

static Diagnostics EvaluateDiagnostics(const Prepared &p, std::span<const double> previous, std::span<const double> current, std::span<const double> next, double psi, double &loss) {
    Diagnostics d{.Psi = psi};
    const auto &c = p.Settings;
    for (std::size_t component = 0; component < p.Components; ++component) {
        const auto a = previous.subspan(component * p.Points, p.Points), b = current.subspan(component * p.Points, p.Points), z = next.subspan(component * p.Points, p.Points);
        auto lap = Laplacian(p, b), lapNext = Laplacian(p, z);
        Vector delta(p.Points), velocity(p.Points);
        for (std::size_t i = 0; i < p.Points; ++i) {
            delta[i] = 0.5 * (z[i] - a[i]) / p.Dt;
            velocity[i] = (b[i] - a[i]) / p.Dt;
            d.Kinetic += 0.5 * p.Mass * p.H * Square((z[i] - b[i]) / p.Dt);
        }
        auto lapVelocity = Laplacian(p, velocity);
        loss += 2 * p.Mass * p.H * p.Dt * (p.Sigma0 * Dot(delta, delta) - (component ? 0 : p.Sigma1 * Dot(delta, lapVelocity)));
        if (c.Split || c.Method == Integrator::Reference) {
            d.LinearPotential -= 0.5 * p.H * (component ? p.LongitudinalTension : c.Tension) * Dot(z, lap);
            if (!component) d.LinearPotential += 0.5 * p.Rigidity * p.H * Dot(lapNext, lap);
        }
    }
    d.NonlinearPotential = 0.5 * Square(psi);
    if (c.Method == Integrator::Reference) {
        if (c.Equation == Model::Cubic) {
            d.NonlinearPotential = 0;
            auto a = Difference(p, current.first(p.Points)), b = Difference(p, next.first(p.Points));
            if (c.Nonlinear)
                for (std::size_t i = 0; i < p.Intervals; ++i)
                    d.NonlinearPotential += p.H * p.NonlinearModulus / 8 * Square(a[i] * b[i]);
        } else if (c.Equation == Model::KirchhoffCarrier) {
            auto lap = Laplacian(p, current);
            d.NonlinearPotential = c.Nonlinear ? p.EA * Square(p.H) / (8 * c.Length) * Square(Dot(next, lap)) : 0;
        } else {
            auto split = p;
            split.Settings.Split = true;
            d.NonlinearPotential = EvaluatePotential(split, Midpoint(current, next), false).Value;
        }
    }
    d.Dissipated = loss;
    d.Energy = d.Kinetic + d.LinearPotential + d.NonlinearPotential + loss;
    return d;
}

Diagnostics Step(const Prepared &p, State &s, bool diagnostics) {
    if (s.Q.size() != p.Points * p.Components || s.Previous.size() != s.Q.size()) throw std::invalid_argument("String state size does not match grid");
    const auto &c = p.Settings;
    auto next = LinearUpdate(p, s, c.Split || c.Method == Integrator::Reference);
    double psiNext{};
    if (c.Method == Integrator::Sav) {
        Vector g;
        if (c.Equation == Model::KirchhoffCarrier && c.Split) {
            g = Laplacian(p, s.Q);
            const double factor = c.Nonlinear ? -std::sqrt(p.EA * Square(p.H) / c.Length) : 0;
            for (double &value : g) value = factor * value;
        } else {
            auto potential = EvaluatePotential(p, s.Q);
            g = std::move(potential.Gradient);
            const double denominator = Psi(p, potential.Value);
            if (denominator == 0 && Dot(g, g) != 0) throw std::runtime_error("Zero SAV denominator with nonzero gradient; use a positive shift");
            for (double &value : g) value = denominator ? value / denominator : 0;
        }
        const double coefficient = Square(p.Dt) / (4 * p.H), b1 = Dot(g, s.Previous);
        Vector chi(next.size());
        for (std::size_t i = 0; i < next.size(); ++i) {
            next[i] = p.InverseMass * (next[i] + coefficient * g[i] * b1 - 4 * coefficient * g[i] * s.Psi);
            chi[i] = p.InverseMass * coefficient * g[i];
        }
        const double correction = Dot(g, next) / (1 + Dot(g, chi));
        double psiDelta{};
        for (std::size_t i = 0; i < next.size(); ++i) {
            next[i] -= chi[i] * correction;
            psiDelta += g[i] * (next[i] - s.Previous[i]);
        }
        psiNext = s.Psi + 0.5 * psiDelta;
    } else if (Geometric(c)) {
        auto split = p;
        split.Settings.Split = true;
        auto potential = EvaluatePotential(split, s.Q);
        for (std::size_t i = 0; i < next.size(); ++i) next[i] = p.InverseMass * (next[i] - Square(p.Dt) / p.H * potential.Gradient[i]);
    } else if (c.Equation == Model::KirchhoffCarrier) {
        auto q = Laplacian(p, s.Q);
        const double factor = c.Nonlinear ? 0.5 * p.Dt * std::sqrt(p.H * p.EA / c.Length) : 0;
        for (double &value : q) value *= factor;
        const double product = Dot(q, s.Previous);
        for (std::size_t i = 0; i < next.size(); ++i) next[i] = p.InverseMass * (next[i] - q[i] * product);
        const double correction = Dot(q, next) / (1 + p.InverseMass * Dot(q, q));
        for (std::size_t i = 0; i < next.size(); ++i) next[i] -= p.InverseMass * q[i] * correction;
    } else {
        // The cubic discrete-gradient reference requires one SPD tridiagonal solve.
        auto q = Difference(p, s.Q);
        const double factor = c.Nonlinear ? Square(p.Dt) * p.NonlinearModulus / (4 * Square(p.H)) : 0;
        Vector diagonal(p.Points), upper(p.Points);
        for (std::size_t i = 0; i < p.Points; ++i) {
            const double left = factor * Square(q[i]), right = factor * Square(q[i + 1]);
            diagonal[i] = 1 / p.InverseMass + left + right;
            upper[i] = -right;
            next[i] += (i ? left * s.Previous[i - 1] : 0) - (left + right) * s.Previous[i] + (i + 1 < p.Points ? right * s.Previous[i + 1] : 0);
            if (i) {
                const double multiplier = -left / diagonal[i - 1];
                diagonal[i] -= multiplier * upper[i - 1];
                next[i] -= multiplier * next[i - 1];
            }
        }
        for (std::size_t i = p.Points; i-- > 0;) next[i] = (next[i] - (i + 1 < p.Points ? upper[i] * next[i + 1] : 0)) / diagonal[i];
    }
    if (c.Method == Integrator::Reference) psiNext = Psi(p, EvaluatePotential(p, Midpoint(s.Q, next), false).Value);
    if (!std::isfinite(psiNext) || !std::ranges::all_of(next, [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("String integration produced non-finite state");
    auto diagnostic = diagnostics || c.Damped ? EvaluateDiagnostics(p, s.Previous, s.Q, next, psiNext, s.Loss) : Diagnostics{.Psi = psiNext};
    s.Previous = std::move(s.Q);
    s.Q = std::move(next);
    s.Psi = psiNext;
    ++s.StepIndex;
    return diagnostic;
}

std::array<double, 2> Pickup(const Prepared &p, std::span<const double> q) {
    if (q.size() != p.Points * p.Components) throw std::invalid_argument("String state size does not match grid");
    return {q[p.Receiver[0]], p.Components == 2 ? q[p.Points + p.Receiver[1]] : 0};
}
} // namespace stiffstring
