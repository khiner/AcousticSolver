#include "Tube.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace tube {
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}
double Positive(const nlohmann::json &j, const char *key, double fallback) {
    const double x = j.value(key, fallback);
    Require(std::isfinite(x) && x > 0, "Physical values must be finite and positive");
    return x;
}
uint32_t Integer(const nlohmann::json &j, const char *key, int fallback, int low, int high) {
    const auto value = j.value(key, nlohmann::json(fallback));
    Require(value.is_number_integer() && value >= low && value <= high, "Integer parameter outside supported range");
    return value.get<uint32_t>();
}
struct Losses {
    double A0;
    std::vector<double> A, B;
};
Losses Coefficients(uint32_t terms) {
    switch (terms) {
#include "TubeLosses.inc"
        default: throw std::invalid_argument("loss_terms must be 0..8 or 16");
    }
}
struct Diffusion {
    struct Memory {
        double Rate, Decay, Energy, Reduced;
    };
    double Sum{};
    std::vector<Memory> Terms;
};
Diffusion PrepareDiffusion(const Losses &losses, double resistance, double storage, double dt, bool lossless) {
    Diffusion result{0, std::vector<Diffusion::Memory>(losses.A.size())};
    if (lossless) return result;
    result.Sum = resistance * losses.A0 / 2;
    for (size_t j = 0; j < losses.A.size(); ++j) {
        const double r = resistance * losses.A[j] / losses.B[j] / 2, c = storage * losses.A[j] / dt;
        const double reduced = r * c / (c + r);
        result.Terms[j] = {r / (c + r), (c - r) / (c + r), c * dt, reduced};
        result.Sum += reduced;
    }
    return result;
}
std::pair<double, double> Legendre(int n, double x) {
    double previous = 1, current = x;
    for (int k = 2; k <= n; ++k) {
        const double next = ((2 * k - 1) * x * current - (k - 1) * previous) / k;
        previous = current;
        current = next;
    }
    return {current, previous};
}
// P coefficients: mass, C0, P->q, P0->q, P->P0, P0->P0, alpha; each memory: Pi->q, Pi->P0, P->Pi, Pi->Pi, Ci.
double AdvancePressure(const Model &t, double *p, const double *v, uint32_t step, uint32_t i) {
    const auto &b = t.Gradient;
    const uint32_t n = t.Elements * t.Order + 1, d = t.Order + 1;
    const uint32_t stride = 2 + t.Terms;
    const double *c = t.PressureCoefficients.data() + i * (7 + 5 * t.Terms);
    double *x = p + i * stride;
    double gradient = 0.;
    const uint32_t first = i ? (i - 1) / t.Order : 0;
    const uint32_t last = i / t.Order < t.Elements ? i / t.Order : t.Elements - 1;
    for (uint32_t e = first; e <= last; ++e)
        for (uint32_t j = 0; j < d; ++j)
            gradient = gradient + b[j * d + i - e * t.Order] * v[(e * d + j) * (1 + t.Terms)];
    double q = c[2] * x[0] + c[3] * x[1];
    for (uint32_t j = 0; j < t.Terms; ++j) q = q + c[7 + 5 * j] * x[2 + j];
    q = q - c[6] * gradient;
    if (i == 0 || i == n - 1) {
        const uint32_t end = i == 0 ? 0 : 1;
        q = t.Ends[2 * end] * q + t.Ends[2 * end + 1] * t.Input[2 * step + end];
    }
    const double next = 2. * q - x[0];
    double p0 = c[4] * (x[0] + next) + c[5] * x[1];
    for (uint32_t j = 0; j < t.Terms; ++j) p0 = p0 + c[8 + 5 * j] * x[2 + j];
    const double delta = x[0] + next - x[1] - p0;
    double e = .5 * (c[0] * next * next + c[1] * p0 * p0);
    for (uint32_t j = 0; j < t.Terms; ++j) {
        x[2 + j] = c[9 + 5 * j] * delta + c[10 + 5 * j] * x[2 + j];
        e = e + .5 * c[11 + 5 * j] * x[2 + j] * x[2 + j];
    }
    x[0] = next;
    x[1] = p0;
    return e;
}

// V coefficients: mass, R0, V->V, gradient->V; each memory: Vi->V, V->Vi, Vi->Vi, Li.
double AdvanceFlow(const Model &t, const double *p, double *v, uint32_t i) {
    const auto &b = t.Gradient;
    const uint32_t d = t.Order + 1, element = i / d, local = i % d;
    const double *c = t.FlowCoefficients.data() + i * (4 + 4 * t.Terms);
    double *x = v + i * (1 + t.Terms);
    double gradient = 0.;
    for (uint32_t j = 0; j < d; ++j) gradient = gradient + b[local * d + j] * p[(element * t.Order + j) * (2 + t.Terms)];
    double next = 0.;
    for (uint32_t j = 0; j < t.Terms; ++j) next = next + c[4 + 4 * j] * x[1 + j];
    next = next + c[3] * gradient + c[2] * x[0];
    const double average = .5 * (next + x[0]);
    double delta = c[1] * average;
    double e = .5 * c[0] * average * average;
    for (uint32_t j = 0; j < t.Terms; ++j) {
        const double old = x[1 + j];
        x[1 + j] = c[5 + 4 * j] * (next + x[0]) + c[6 + 4 * j] * old;
        e = e + .25 * c[7 + 4 * j] * (x[1 + j] * x[1 + j] + old * old);
        delta = delta + c[7 + 4 * j] * (x[1 + j] - old) / t.Dt;
    }
    x[0] = next;
    return e + t.Dt * t.Dt * .125 * (delta * delta - gradient * gradient) / c[0];
}

} // namespace

Model Prepare(const nlohmann::json &config) {
    Model t;
    t.Order = Integer(config, "order", 3, 1, 10);
    t.Terms = Integer(config, "loss_terms", 8, 0, 16);
    const auto losses = Coefficients(t.Terms);
    const bool lossless = config.value("lossless", false);
    const double rate = Integer(config, "sample_rate", 192000, 8000, 2000000);
    const double duration = Positive(config, "seconds", .25);
    Require(duration * rate <= 20000000 && duration * rate >= 1, "Render duration outside supported range");
    t.Steps = uint32_t(std::floor(duration * rate));
    t.Dt = 1 / rate;
    const auto bore = config.at("bore").get<std::vector<std::array<double, 2>>>();
    Require(bore.size() >= 2 && bore.size() <= 256 && bore.front()[0] == 0, "bore requires [position, radius] pairs starting at zero");
    for (size_t i = 0; i < bore.size(); ++i)
        Require(std::isfinite(bore[i][0]) && std::isfinite(bore[i][1]) && bore[i][1] > 0 && (i == 0 || bore[i][0] > bore[i - 1][0]), "Bore positions must increase and radii must be positive");
    const auto subdivisions = config.value("elements_per_segment", std::vector<uint32_t>(bore.size() - 1, 8));
    Require(subdivisions.size() == bore.size() - 1, "elements_per_segment must match bore segments");
    std::vector<double> edges{0};
    for (size_t i = 0; i < subdivisions.size(); ++i) {
        Require(subdivisions[i] >= 1 && subdivisions[i] <= 512, "Each bore segment requires 1..512 elements");
        for (uint32_t e = 1; e <= subdivisions[i]; ++e)
            edges.push_back(std::lerp(bore[i][0], bore[i + 1][0], double(e) / subdivisions[i]));
    }
    t.Elements = uint32_t(edges.size() - 1);
    const uint32_t d = t.Order + 1, np = t.Elements * t.Order + 1, nv = t.Elements * d;
    Require(np <= 4096 && nv <= 8192, "Tube mesh exceeds 4096 pressure or 8192 flow nodes");
    const auto air = config.value("air", nlohmann::json::object());
    const double rho = Positive(air, "density", 1.2041), speed = Positive(air, "sound_speed", 343.2);
    const double mu = Positive(air, "viscosity", 1.825e-5), kappa = Positive(air, "conductivity", .0257);
    const double cp = Positive(air, "heat_capacity", 1005), gamma = Positive(air, "heat_ratio", 1.4);
    Require(gamma > 1, "heat_ratio must exceed one");
    std::vector<double> nodes(d), weights(d), barycentric(d, 1);
    for (uint32_t i = 0; i < d; ++i) {
        double x = -std::cos(std::numbers::pi * i / t.Order);
        if (i && i < t.Order)
            for (int k = 0; k < 30; ++k) {
                auto [pn, previous] = Legendre(t.Order, x);
                const double delta = (x * pn - previous) / ((t.Order + 1) * pn);
                x -= delta;
                if (std::abs(delta) < 1e-15) break;
            }
        nodes[i] = .5 * (1 + x);
        const double pn = Legendre(t.Order, x).first;
        weights[i] = 1 / (t.Order * (t.Order + 1) * pn * pn);
    }
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j)
            if (j != i) barycentric[i] /= nodes[i] - nodes[j];
    t.Gradient.resize(d * d);
    for (uint32_t i = 0; i < d; ++i) {
        double diagonal = 0;
        for (uint32_t j = 0; j < d; ++j)
            if (i != j) {
                const double derivative = barycentric[j] / (barycentric[i] * (nodes[i] - nodes[j]));
                t.Gradient[i * d + j] = -weights[i] * derivative;
                diagonal -= derivative;
            }
        t.Gradient[i * d + i] = -weights[i] * diagonal;
    }
    const auto radius = [&](double x) {
        const auto it = std::upper_bound(bore.begin(), bore.end(), x, [](double a, const auto &b) { return a < b[0]; });
        const auto i = std::clamp<size_t>(it - bore.begin(), 1, bore.size() - 1);
        return std::lerp(bore[i - 1][1], bore[i][1], (x - bore[i - 1][0]) / (bore[i][0] - bore[i - 1][0]));
    };
    std::vector<double> mp(np), mv(nv), g(np), r(nv);
    t.Positions.resize(np);
    for (uint32_t e = 0; e < t.Elements; ++e)
        for (uint32_t j = 0; j < d; ++j) {
            const double x = std::lerp(edges[e], edges[e + 1], nodes[j]), w = (edges[e + 1] - edges[e]) * weights[j];
            const double area = std::numbers::pi * radius(x) * radius(x);
            const uint32_t p = e * t.Order + j, v = e * d + j;
            t.Positions[p] = x;
            mp[p] += w * area / (rho * speed * speed);
            mv[v] = w * rho / area;
            g[p] += w * std::numbers::pi * kappa * (gamma - 1) / (rho * rho * speed * speed * cp);
            r[v] = w * std::numbers::pi * mu / (area * area);
        }
    // A row-sum bound on M_p^-1 B^T M_v^-1 B gives a conservative CFL limit.
    std::vector<double> rowBound(np);
    for (uint32_t e = 0; e < t.Elements; ++e)
        for (uint32_t i = 0; i < d; ++i) {
            double sum = 0;
            for (uint32_t j = 0; j < d; ++j) sum += std::abs(t.Gradient[i * d + j]);
            for (uint32_t j = 0; j < d; ++j) rowBound[e * t.Order + j] += std::abs(t.Gradient[i * d + j]) * sum / mv[e * d + i];
        }
    double bound = 0;
    for (uint32_t i = 0; i < np; ++i) bound = std::max(bound, rowBound[i] / mp[i]);
    const double minimumRate = .5 * std::sqrt(bound);
    if (rate <= minimumRate) throw std::invalid_argument("sample_rate must exceed conservative CFL bound " + std::to_string(minimumRate) + " Hz; raise rate or reduce mesh/order");
    t.PressureCoefficients.reserve(np * (7 + 5 * t.Terms));
    t.FlowCoefficients.reserve(nv * (4 + 4 * t.Terms));
    for (uint32_t i = 0; i < np; ++i) {
        const double c0 = lossless ? 0 : mp[i] * (gamma - 1), mass = mp[i] / t.Dt;
        const auto [sum, memory] = PrepareDiffusion(losses, g[i], c0, t.Dt, lossless);
        const double denominator0 = c0 / t.Dt + sum;
        const double coupling = lossless ? 0 : (c0 / t.Dt) * sum / denominator0;
        const double denominator = mass + coupling;
        t.PressureCoefficients.insert(t.PressureCoefficients.end(), {mp[i], c0, mass / denominator, coupling / denominator, lossless ? 0 : sum / denominator0, lossless ? 0 : (c0 / t.Dt - sum) / denominator0, 1 / (2 * denominator)});
        for (const auto &m : memory)
            t.PressureCoefficients.insert(t.PressureCoefficients.end(), {lossless ? 0 : m.Reduced / sum * (coupling / denominator), lossless ? 0 : -2 * m.Reduced / denominator0, m.Rate, m.Decay, m.Energy});
    }
    for (uint32_t i = 0; i < nv; ++i) {
        const double mass = mv[i] / t.Dt;
        const auto [sum, memory] = PrepareDiffusion(losses, r[i], mv[i], t.Dt, lossless);
        t.FlowCoefficients.insert(t.FlowCoefficients.end(), {mv[i], lossless ? 0 : r[i] * losses.A0, (mass - sum) / (mass + sum), 1 / (mass + sum)});
        for (const auto &m : memory)
            t.FlowCoefficients.insert(t.FlowCoefficients.end(), {2 * m.Reduced / (mass + sum), m.Rate, m.Decay, m.Energy});
    }
    const auto boundaries = config.value("ends", std::vector<nlohmann::json>{"closed", "open"});
    Require(boundaries.size() == 2, "ends must have two boundary conditions");
    for (uint32_t end = 0; end < 2; ++end) {
        const double alpha = t.PressureCoefficients[(end ? np - 1 : 0) * (7 + 5 * t.Terms) + 6];
        double impedance = 0;
        if (boundaries[end] == "closed") impedance = INFINITY;
        else if (boundaries[end] == "matched") impedance = rho * speed / (std::numbers::pi * bore[end ? bore.size() - 1 : 0][1] * bore[end ? bore.size() - 1 : 0][1]);
        else if (boundaries[end].is_number()) {
            impedance = boundaries[end].get<double>();
            Require(std::isfinite(impedance) && impedance >= 0, "Endpoint impedance must be finite and nonnegative");
        } else Require(boundaries[end] == "open", "Unknown endpoint condition");
        const double factor = std::isinf(impedance) ? 1 : impedance / (impedance + alpha);
        t.Ends.insert(t.Ends.end(), {factor, factor * alpha});
    }
    const auto pickups = config.value("receivers", std::vector<double>{0, bore.back()[0] / 2});
    Require(!pickups.empty() && pickups.size() <= 32, "Require 1..32 receiver positions");
    t.Channels = uint32_t(pickups.size());
    Require(size_t(t.Steps) * (t.Channels + 3) <= 32 * 1024 * 1024, "Trace storage exceeds 256 MiB");
    t.Pickups.resize(pickups.size() * np);
    for (size_t ch = 0; ch < pickups.size(); ++ch) {
        const double x = pickups[ch];
        Require(std::isfinite(x) && x >= 0 && x <= bore.back()[0], "Receiver outside tube");
        const size_t e = std::min<size_t>(std::upper_bound(edges.begin(), edges.end(), x) - edges.begin() - 1, t.Elements - 1);
        const double local = (x - edges[e]) / (edges[e + 1] - edges[e]);
        for (uint32_t j = 0; j < d; ++j) {
            double weight = 1;
            for (uint32_t k = 0; k < d; ++k)
                if (k != j) weight *= (local - nodes[k]) / (nodes[j] - nodes[k]);
            t.Pickups[ch * np + e * t.Order + j] = weight;
        }
    }
    t.Input.resize(2 * t.Steps);
    const auto pulses = config.value("pulses", nlohmann::json::array({{{"start", 0}, {"duration", .001}, {"flow", 1e-6}, {"end", 0}}}));
    Require(pulses.is_array() && pulses.size() <= 1024, "pulses must be an array of at most 1024 sources");
    double sourceEnd = 0;
    for (const auto &pulse : pulses) {
        const double start = pulse.value("start", 0.), length = Positive(pulse, "duration", .001), flow = pulse.value("flow", 1e-6);
        const uint32_t end = Integer(pulse, "end", 0, 0, 1);
        Require(std::isfinite(start) && start >= 0 && std::isfinite(flow), "Invalid pulse start or flow");
        Require(t.Ends[2 * end] > 0, "Cannot inject volume flow at a prescribed zero-pressure end");
        sourceEnd = std::max(sourceEnd, start + length);
        for (uint32_t step = 0; step < t.Steps; ++step) {
            const double time = (step + .5) * t.Dt - start;
            if (time >= 0 && time < length) t.Input[2 * step + end] += .5 * flow * (1 - std::cos(2 * std::numbers::pi * time / length));
        }
    }
    for (const auto *v : {&t.PressureCoefficients, &t.FlowCoefficients, &t.Input})
        for (const double x : *v) Require(std::isfinite(x), "Nonfinite prepared coefficient or excitation");
    t.Metadata = {{"config", config}, {"sample_rate", rate}, {"steps", t.Steps}, {"order", t.Order}, {"elements", t.Elements}, {"pressure_nodes", np}, {"flow_nodes", nv}, {"loss_terms", t.Terms}, {"lossless", lossless}, {"minimum_sample_rate", minimumRate}, {"source_end", sourceEnd}, {"bore", bore}, {"element_edges", edges}, {"receivers", pickups}, {"air", {{"density", rho}, {"sound_speed", speed}, {"viscosity", mu}, {"conductivity", kappa}, {"heat_capacity", cp}, {"heat_ratio", gamma}}}};
    return t;
}

Trace Run(const Model &t) {
    const uint32_t np = t.Elements * t.Order + 1, nv = t.Elements * (t.Order + 1);
    std::vector<double> p(np * (2 + t.Terms)), v(nv * (1 + t.Terms));
    Trace trace;
    trace.Samples.resize(size_t(t.Steps) * t.Channels);
    trace.Energy.resize(t.Steps);
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t step = 0; step < t.Steps; ++step) {
        for (uint32_t i = 0; i < np; ++i) trace.Energy[step] += AdvancePressure(t, p.data(), v.data(), step, i);
        for (uint32_t i = 0; i < nv; ++i) trace.Energy[step] += AdvanceFlow(t, p.data(), v.data(), i);
        for (uint32_t ch = 0; ch < t.Channels; ++ch)
            for (uint32_t i = 0; i < np; ++i) trace.Samples[size_t(step) * t.Channels + ch] += t.Pickups[ch * np + i] * p[i * (2 + t.Terms)];
    }
    trace.Seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    trace.State = std::move(p);
    trace.State.insert(trace.State.end(), v.begin(), v.end());
    return trace;
}
} // namespace tube
