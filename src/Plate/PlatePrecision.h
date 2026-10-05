#pragma once
#include "Plate.h"
#include "PlateReal.h"
#include <boost/multiprecision/mpfr.hpp>

namespace plate {
// Setup arithmetic has guard bits beyond the GPU's 256-bit significand.
using Precise = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<128>>;
inline PlateReal Pack(Precise value) {
    PlateReal out;
    if (value == 0) return out;
    out.Sign = value < 0 ? -1 : 1;
    value = abs(value);
    int exponent;
    value = frexp(value, &exponent);
    out.Exponent = exponent - 1;
    for (int i = PlateReal::Limbs - 1; i >= 0; --i) {
        value = ldexp(value, 32);
        out.Word[i] = value.convert_to<PlateWord>();
        value -= out.Word[i];
    }
    return out;
}
inline std::vector<PlateReal> Pack(const std::vector<Precise> &values) {
    std::vector<PlateReal> out;
    out.reserve(values.size());
    for (const auto &value : values) out.push_back(Pack(value));
    return out;
}
struct PreciseTables {
    Precise Dt, Scale;
    std::vector<Precise> Kx, Ky, Omega2, Sigma, Sx, Sy, Cx, Cy;
    explicit PreciseTables(const Prepared &p, bool denseTransforms) {
        const Precise pi = acos(Precise(-1)), root = sqrt(Precise(p.Settings.Ratio));
        Dt = Precise(1) / p.Settings.SampleRate;
        Scale = sqrt(Precise(p.Nx) * p.Ny);
        for (size_t i = 0; i <= p.Mx; ++i) Kx.push_back(Precise(i) * pi / root);
        for (size_t i = 0; i <= p.My; ++i) Ky.push_back(Precise(i) * pi * root);
        for (size_t x = 1; x <= p.Mx; ++x)
            for (size_t y = 1; y <= p.My; ++y) {
                Precise k2 = Kx[x] * Kx[x] + Ky[y] * Ky[y], w = Precise(p.Settings.Kappa) * k2, s = Precise(p.Settings.Sigma0) + Precise(p.Settings.Sigma1) * k2;
                if (p.Settings.UseExact) {
                    Precise e = exp(-s * Dt), phase2 = w * w - s * s;
                    Precise const oscillation = phase2 >= 0 ? Precise(cos(sqrt(phase2) * Dt)) : Precise(cosh(sqrt(-phase2) * Dt));
                    Omega2.emplace_back(2 / (Dt * Dt) * (1 - 2 * e * oscillation + e * e) / (1 + e * e));
                    Sigma.emplace_back(tanh(s * Dt) / Dt);
                } else {
                    Omega2.emplace_back(w * w);
                    Sigma.push_back(s);
                }
            }
        auto const table = [&](size_t modes, size_t nodes, bool cosine) {
            std::vector<Precise> values;
            for (size_t i = 0; i < modes + size_t(cosine); ++i)
                for (size_t j = 0; j < nodes; ++j) {
                    Precise const phase = pi * Precise(i + size_t(!cosine)) * (Precise(j) + Precise(.5)) / nodes;
                    values.push_back(sqrt(Precise(cosine && !i ? 1 : 2) / nodes) * (cosine ? Precise(cos(phase)) : Precise(sin(phase))));
                }
            return values;
        };
        if (!denseTransforms) return;
        Sx = table(p.Mx, p.Nx, false);
        Sy = table(p.My, p.Ny, false);
        Cx = table(p.Mx, p.Nx, true);
        Cy = table(p.My, p.Ny, true);
    }
};
inline std::vector<Precise> PrecisePointBasis(const Prepared &p, double x, double y) {
    ValidatePoint(p, x, y);
    const Precise pi = acos(Precise(-1)), root = sqrt(Precise(p.Settings.Ratio));
    std::vector<Precise> out(p.Mx * p.My);
    for (size_t i = 0; i < p.Mx; ++i)
        for (size_t j = 0; j < p.My; ++j)
            if (p.Active[i * p.My + j]) out[i * p.My + j] = 2 * sin(Precise(i + 1) * pi / root * Precise(x)) * sin(Precise(j + 1) * pi * root * Precise(y));
    return out;
}
struct PreciseStrike {
    std::vector<Precise> Basis;
    Precise Amplitude, Start, Duration;
    int Type;
};
inline std::vector<PlateReal> PreciseForces(const Prepared &p, const std::vector<PreciseStrike> &strikes, size_t offset, size_t count, size_t cycle) {
    std::vector<PlateReal> result;
    const Precise pi = acos(Precise(-1)), dt = Precise(1) / p.Settings.SampleRate;
    for (size_t n = 0; n < count; ++n) {
        const Precise time = Precise((offset + n) % cycle) * dt;
        std::vector<Precise> force;
        for (const auto &s : strikes)
            if (time >= s.Start && time <= s.Start + s.Duration) {
                const Precise amplitude = Precise(.5) * s.Amplitude * (1 - cos(Precise(s.Type) * pi * (time - s.Start) / s.Duration));
                if (amplitude == 0) continue;
                if (force.empty()) force.resize(p.Mx * p.My);
                for (size_t i = 0; i < force.size(); ++i) force[i] += s.Basis[i] * amplitude;
            }
        if (!force.empty()) {
            if (result.empty()) result.resize(count * p.Mx * p.My);
            for (size_t i = 0; i < force.size(); ++i) result[n * force.size() + i] = Pack(force[i]);
        }
    }
    return result;
}
} // namespace plate
