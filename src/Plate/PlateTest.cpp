#include "Plate.h"
#include "PlateGpu.h"
#include "PlateMetal.h"
#include "PlatePrecision.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
using namespace plate;
using json = nlohmann::json;
json Results = json::object();
int Failures = 0;

void Check(const std::string &name, double error, double tolerance) {
    const bool pass = std::isfinite(error) && error <= tolerance;
    Results[name] = {{"error", error}, {"tolerance", tolerance}, {"pass", pass}};
    std::cout << (pass ? "PASS " : "FAIL ") << name << ": " << error << " <= " << tolerance << '\n';
    Failures += !pass;
}

double Norm(std::span<const double> v) {
    return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.));
}

double Relative(std::span<const double> actual, std::span<const double> expected, double floor = 1e-30) {
    if (actual.size() != expected.size()) throw std::runtime_error("Comparison shape mismatch");
    double error = 0;
    for (size_t i = 0; i < actual.size(); ++i) error += std::pow(actual[i] - expected[i], 2);
    return std::sqrt(error) / std::max(Norm(expected), floor);
}

void ArithmeticChecks(bool metal) {
    constexpr uint32_t count = 2048, operations = 6;
    std::vector<PlateReal> input(2 * count + 1), output(operations * count + 3);
    for (size_t i = 0; i < count; ++i) {
        const Precise a = pow(Precise(10), int(i % 25) - 12) * (Precise("1.123456789") + sin(Precise(i)));
        const Precise b = pow(Precise(10), int((i * 7) % 25) - 12) * (Precise("1.23456789") + cos(Precise(i)));
        input[2 * i] = Pack(a);
        input[2 * i + 1] = Pack(i % 3 ? b : Precise(-b));
    }
    // Retain cancellation cases near a binary32 rounding boundary.
    for (size_t i = 96; i < 128; ++i) {
        const int exponent = 2 * int(i - 96) - 32;
        const double midpoint = 1. + std::ldexp(1., -24);
        input[2 * i] = MakeReal(std::ldexp(midpoint, exponent));
        input[2 * i + 1] = MakeReal(std::ldexp(midpoint + (i % 2 ? 1 : -1) * std::ldexp(1., -47), exponent));
    }
    input.back() = MakeReal(1e-9);
    const auto special = [&] {
        output[operations * count] = (PlateReal(1.f) + MakeReal(std::ldexp(1., -30))) - PlateReal(1.f);
        PlateReal accumulator(1.f);
        for (int i = 0; i < 100000; ++i) accumulator += MakeReal(1e-9);
        output[operations * count + 1] = accumulator;
        accumulator = PlateReal(0.f);
        for (size_t i = 0; i < count; ++i) accumulator += input[2 * i] * input[2 * i + 1];
        output[operations * count + 2] = accumulator;
    };
    if (!metal) {
        for (size_t i = 0; i < count; ++i) {
            const auto a = input[2 * i], b = input[2 * i + 1];
            output[operations * i] = a + b;
            output[operations * i + 1] = a - b;
            output[operations * i + 2] = a * b;
            output[operations * i + 3] = a / b;
            output[operations * i + 4] = PlateSqrt(a);
            output[operations * i + 5] = (a + b) - a;
        }
        special();
    } else {
        const auto source = metal::Read(ACOUSTIC_PROJECT_ROOT "/src/Plate/PlateReal.h") + R"(
kernel void PlateArithmeticProbe(device const PlateReal *input [[buffer(0)]],
                                 device PlateReal *output [[buffer(1)]],
                                 constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    PlateReal a=input[2*i], b=input[2*i+1];
    output[6*i]=a+b; output[6*i+1]=a-b; output[6*i+2]=a*b;
    output[6*i+3]=a/b; output[6*i+4]=PlateSqrt(a); output[6*i+5]=(a+b)-a;
    if (i==0) {
        output[6*count]=(PlateReal(1.f)+PlateReal(0x1p-30f))-PlateReal(1.f);
        PlateReal accumulator(1.f), increment=input[2*count];
        for(uint j=0;j<100000;++j) accumulator+=increment;
        output[6*count+1]=accumulator;
        accumulator=PlateReal(0.f);
        for(uint j=0;j<count;++j) accumulator+=input[2*j]*input[2*j+1];
        output[6*count+2]=accumulator;
    }
})";
        auto &context = MetalContext::Get();
        const auto library = metal::Compile(source);
        const auto pipeline = metal::Pipeline(library.get(), "PlateArithmeticProbe");
        GpuBuffer arguments, results;
        metal::Upload(arguments, input);
        results.Resize(output.size() * sizeof(PlateReal));
        context.Dispatch(pipeline.get(), {(count + 127) / 128}, {128}, {&arguments, &results}, &count, sizeof(count));
        const auto *values = results.As<PlateReal>();
        std::copy_n(values, output.size(), output.begin());
    }
    const auto unpack = [](PlateReal v) {
        Precise value = 0;
        for (int i = PlateReal::Limbs - 1; i >= 0; --i) value = ldexp(value, 32) + v.Word[i];
        return Precise(v.Sign * ldexp(value, v.Exponent - 255));
    };
    double precisionError = 0;
    for (size_t i = 0; i < count; ++i) {
        const Precise a = unpack(input[2 * i]), b = unpack(input[2 * i + 1]);
        const Precise expected[]{a + b, a - b, a * b, a / b, sqrt(a), b};
        for (size_t op = 0; op < operations; ++op) {
            Precise const scale = (op < 2 || op == 5) ? Precise(max(abs(a), abs(b))) : Precise(abs(expected[op]));
            precisionError = std::max(precisionError, double(abs(unpack(output[operations * i + op]) - expected[op]) / scale));
        }
    }
    Check(metal ? "wide_metal_mpfr" : "wide_host_mpfr", precisionError, 1e-70);
    std::array<double, operations> errors{};
    long double dot = 0;
    for (size_t i = 0; i < count; ++i) {
        const double a = ToDouble(input[2 * i]), b = ToDouble(input[2 * i + 1]);
        const double expected[]{a + b, a - b, a * b, a / b, std::sqrt(a), b};
        for (size_t op = 0; op < operations; ++op) {
            const double scale = (op < 2 || op == 5) ? std::max(std::abs(a), std::abs(b)) : std::abs(expected[op]);
            errors[op] = std::max(errors[op], std::abs(ToDouble(output[operations * i + op]) - expected[op]) / scale);
        }
        dot += static_cast<long double>(a) * b;
    }
    const std::string prefix = metal ? "wide_metal_" : "wide_host_";
    double cancellation = 0;
    for (size_t i = 96; i < 128; ++i) {
        const double expected = ToDouble(input[2 * i]) - ToDouble(input[2 * i + 1]);
        cancellation = std::max(cancellation, std::abs((ToDouble(output[operations * i + 1]) - expected) / expected));
    }
    Check(prefix + "near_cancellation", cancellation, 0);
    const char *names[]{"addition", "subtraction", "multiplication", "division", "sqrt", "cancellation"};
    for (size_t op = 0; op < operations; ++op) Check(prefix + names[op], errors[op], 5e-13);
    Check(prefix + "exact_small_increment", std::abs(ToDouble(output[operations * count]) - std::ldexp(1., -30)), 0);
    Check(prefix + "repeated_tiny_increments", std::abs(ToDouble(output[operations * count + 1]) - 1.0001), 1e-11);
    Check(prefix + "dot_product", std::abs((ToDouble(output[operations * count + 2]) - double(dot)) / double(dot)), 5e-13);
}

Config Small() {
    Config c;
    c.SampleRate = 16000;
    c.CutoffHz = 2400;
    c.Kappa = 60;
    c.Sigma0 = 0;
    c.Sigma1 = 0;
    return c;
}

std::vector<double> FreeDisplacement(double rate, double cutoff, double amplitude) {
    auto c = Small();
    c.SampleRate = rate;
    c.CutoffHz = cutoff;
    // The unregulated SAV method isolates convergence from sign(velocity)'s
    // roundoff-sensitive decisions in symmetry modes with exactly zero motion.
    c.Lambda = 0;
    const auto p = Prepare(c);
    std::vector<double> q(p.Mx * p.My), previous(q.size());
    q[0] = amplitude;
    const auto potential = EvaluatePotential(p, q);
    // Taylor initialization imposes the same physical zero initial velocity
    // at each time step size, rather than shifting the initial clock by dt/2.
    for (size_t i = 0; i < q.size(); ++i) {
        const double beta2 = std::pow(p.Kx[i / p.My], 2) + std::pow(p.Ky[i % p.My], 2);
        previous[i] = q[i] - .5 * p.Dt * p.Dt * c.Kappa * c.Kappa * (beta2 * beta2 * q[i] + potential.Gradient[i]);
    }
    auto state = InitialState(p, q, previous);
    for (size_t i = 0; i < size_t(.016 * rate); ++i) Step(p, state);
    return state.Q;
}

void BackendChecks() {
    // Representative measured workloads; these are intentionally not a copy
    // of the cost formula. Different FFT padding can change the winning path.
    struct Example {
        double radius, ratio;
        Backend expected;
    };
    const Example examples[]{
        {10, 1, Backend::MetalDense}, {22, 4, Backend::MetalDense}, {42, 1, Backend::MetalDense}, {110, 1, Backend::MetalDense}, {170, 1, Backend::MetalFft}, {84, .25, Backend::MetalFft}, {84, 4, Backend::MetalFft}
    };
    for (size_t i = 0; i < std::size(examples); ++i) {
        Config c;
        c.CutoffHz = 17000;
        c.Ratio = examples[i].ratio;
        c.Kappa = 34000 / (std::numbers::pi * (std::pow(examples[i].radius + .5, 2) + 1));
        Check("backend_choice_" + std::to_string(i), SelectBackend(Prepare(c)) == examples[i].expected ? 0 : 1, 0);
    }
    Config c;
    c.Nonlinear = false;
    Check("backend_linear_cpu", SelectBackend(Prepare(c)) == Backend::Cpu ? 0 : 1, 0);
}

void CpuChecks() {
    BackendChecks();
    ArithmeticChecks(false);
    const auto c = Small();
    const auto p = Prepare(c);
    const size_t modes = p.Mx * p.My;
    std::vector<double> q(modes);
    for (size_t i = 0; i < modes; ++i)
        if (p.Active[i]) q[i] = .03 * std::sin(double(i) + .5);
    const auto potential = EvaluatePotential(p, q);
    const double euler = std::inner_product(q.begin(), q.end(), potential.Gradient.begin(), 0.);
    Check("quartic_potential_identity", std::abs(euler - 4 * potential.Value) / (4 * potential.Value), 2e-12);
    double gradient_error = 0;
    for (size_t i = 0; i < modes; ++i)
        if (p.Active[i]) {
            constexpr double delta = 1e-7;
            auto perturbed = q;
            perturbed[i] += delta;
            const double plus = EvaluatePotential(p, perturbed, false).Value;
            perturbed[i] -= 2 * delta;
            const double minus = EvaluatePotential(p, perturbed, false).Value;
            gradient_error += std::pow((plus - minus) / (2 * delta) - potential.Gradient[i], 2);
        }
    Check("potential_finite_difference_gradient", std::sqrt(gradient_error) / Norm(potential.Gradient), 2e-8);
    Check("zero_potential", EvaluatePotential(p, std::vector<double>(modes)).Value, 0);
    auto padded_config = c;
    padded_config.Oversampling = 2.;
    const auto padded = EvaluatePotential(Prepare(padded_config), q);
    Check("dealiasing_padding_gradient", Relative(potential.Gradient, padded.Gradient), 2e-12);
    Check("dealiasing_padding_potential", std::abs(potential.Value / padded.Value - 1), 2e-12);
    // An orthonormal one-dimensional transform must retain inner products.
    double orthogonality = 0;
    for (bool const cosine : {false, true}) {
        const auto &table = cosine ? p.Cx : p.Sx;
        const size_t rows = p.Mx + size_t(cosine);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < rows; ++j) {
                double value = 0;
                for (size_t k = 0; k < p.Nx; ++k) value += table[i * p.Nx + k] * table[j * p.Nx + k];
                orthogonality = std::max(orthogonality, std::abs(value - (i == j ? 1. : 0.)));
            }
    }
    Check("transform_orthogonality", orthogonality, 2e-14);
    const auto coarse = FreeDisplacement(8000, 2400, .2), medium = FreeDisplacement(16000, 2400, .2);
    const auto fine = FreeDisplacement(32000, 2400, .2), finest = FreeDisplacement(64000, 2400, .2);
    Check("time_step_convergence_medium_over_coarse", Relative(medium, finest) / Relative(coarse, finest), .4);
    Check("time_step_convergence_fine_over_medium", Relative(fine, finest) / Relative(medium, finest), .4);
    const double cutoff_low = FreeDisplacement(32000, 1200, .4)[0], cutoff_mid = FreeDisplacement(32000, 2400, .4)[0];
    const double cutoff_high = FreeDisplacement(32000, 4800, .4)[0], cutoff_reference = FreeDisplacement(32000, 6000, .4)[0];
    Check("spectral_cutoff_convergence", std::abs((cutoff_mid - cutoff_reference) / (cutoff_low - cutoff_reference)), .2);
    Check("spectral_cutoff_converged_fundamental", std::abs((cutoff_high - cutoff_reference) / cutoff_reference), 1e-10);

    auto linear_config = c;
    linear_config.Nonlinear = false;
    linear_config.Sigma0 = 5;
    const auto linear = Prepare(linear_config);
    auto state = InitialState(linear, .02);
    const double beta2 = std::numbers::pi * std::numbers::pi * (1 / c.Ratio + c.Ratio);
    const double omega = c.Kappa * beta2, decay = std::exp(-5 * linear.Dt);
    const double theta = std::sqrt(omega * omega - 25) * linear.Dt;
    const double sine_weight = .02 * (std::cos(theta) - decay) / std::sin(theta);
    double linear_error = 0;
    for (size_t n = 1; n <= 512; ++n) {
        Step(linear, state);
        const double exact = std::pow(decay, double(n)) * (.02 * std::cos(n * theta) + sine_weight * std::sin(n * theta));
        linear_error = std::max(linear_error, std::abs(state.Q[0] - exact) / .02);
    }
    Check("linear_exact_frequency_and_decay", linear_error, 2e-11);

    state = InitialState(p, q, q);
    const double initial_energy = Energy(p, state);
    double conservation = 0;
    for (size_t n = 0; n < 1024; ++n) {
        Step(p, state);
        conservation = std::max(conservation, std::abs(Energy(p, state) - initial_energy) / initial_energy);
    }
    Check("nonlinear_conservative_energy", conservation, 2e-10);

    auto forced_config = c;
    forced_config.Sigma0 = 2;
    forced_config.Sigma1 = 1e-4;
    const auto forced = Prepare(forced_config);
    const auto source = PointBasis(forced, .17, .42);
    state = InitialState(forced);
    const double initial = Energy(forced, state);
    double work = 0, max_energy = 0, balance = 0, max_psi = 0, max_drift = 0;
    std::vector<double> force(modes);
    for (size_t n = 0; n < 2048; ++n) {
        double amplitude = 0;
        for (const double start : {0., .03, .06}) amplitude += RaisedCosineStrike(n * forced.Dt, start, .002, 2e5);
        for (size_t i = 0; i < modes; ++i) force[i] = source[i] * amplitude;
        const auto d = Step(forced, state, force);
        work += forced.Dt * (d.InputPower - d.DissipatedPower);
        const double e = Energy(forced, state);
        max_energy = std::max(max_energy, e);
        balance = std::max(balance, std::abs(e - initial - work));
        max_psi = std::max(max_psi, std::abs(d.Psi));
        max_drift = std::max(max_drift, std::abs(d.Drift));
    }
    Check("forced_damped_energy_work_balance", balance / max_energy, 2e-10);
    Check("repeated_strike_sav_drift", max_drift / std::max(max_psi, 1e-12), .1);
    state = InitialState(p);
    for (int i = 0; i < 16; ++i) Step(p, state);
    Check("zero_state_is_stationary", Norm(state.Q), 0);
    int accepted_invalid = 0;
    for (int i = 0; i < 6; ++i) {
        auto invalid = c;
        if (i == 0) invalid.SampleRate = 0;
        if (i == 1) invalid.Kappa = -1;
        if (i == 2) invalid.Ratio = 0;
        if (i == 3) invalid.CutoffHz = c.SampleRate;
        if (i == 4) invalid.Sigma0 = -1;
        if (i == 5) invalid.Epsilon = std::numeric_limits<double>::quiet_NaN();
        try {
            Prepare(invalid);
            ++accepted_invalid;
        } catch (const std::exception &) {}
    }
    Check("invalid_configuration_rejected", accepted_invalid, 0);
    Check("half_cosine_inclusive_endpoint", std::abs(RaisedCosineStrike(.032, .03, .002, 100, 1) - 100), 1e-12);
    Check("repeated_half_cosine_endpoint", std::abs(RepeatedCosineStrike(132, 100, 1000, .03, .002, 100, 1) - 100), 1e-12);
    Check("raised_cosine_endpoint", std::abs(RaisedCosineStrike(.032, .03, .002, 100, 2)), 1e-12);
    auto automatic = Small();
    automatic.CutoffHz = 0;
    Check("automatic_exact_cutoff", std::abs(Prepare(automatic).Settings.CutoffHz - automatic.SampleRate / 2), 0);
    automatic.UseExact = false;
    Check("automatic_standard_cutoff", std::abs(Prepare(automatic).Settings.CutoffHz - automatic.SampleRate / std::numbers::pi), 0);
}

void GpuChecks(TransformBackend backend = TransformBackend::Dense) {
    const std::string prefix = backend == TransformBackend::Fft ? "fft_" : "";
    if (backend == TransformBackend::Dense) ArithmeticChecks(true);
    auto c = Small();
    c.Sigma0 = 2;
    const auto p = Prepare(c);
    const size_t modes = p.Mx * p.My;
    Gpu gpu(p, backend);
    std::vector<double> q(modes);
    for (size_t i = 0; i < modes; ++i)
        if (p.Active[i]) q[i] = .03 * std::sin(double(i) + .5);
    const auto truth = EvaluatePotential(p, q), actual = gpu.EvaluatePotential(q);
    Check(prefix + "metal_potential", std::abs(actual.Value / truth.Value - 1), 1e-11);
    Check(prefix + "metal_potential_gradient", Relative(actual.Gradient, truth.Gradient), 1e-11);
    Check(prefix + "metal_airy", Relative(actual.Airy, truth.Airy), 1e-11);
    auto state = InitialState(p);
    gpu.Reset(state);
    const auto pickup1 = PointBasis(p, .64, .79), pickup2 = PointBasis(p, .3, .6), source = PointBasis(p, .17, .42);
    auto pickup = pickup1;
    pickup.insert(pickup.end(), pickup2.begin(), pickup2.end());
    constexpr size_t steps = 1024;
    std::vector<double> forces(steps * modes), samples(steps * 2), states(steps * modes);
    std::vector<Diagnostics> diagnostics;
    for (size_t n = 0; n < steps; ++n) {
        const double amplitude = RaisedCosineStrike(n * p.Dt, 0, .002, 2e5);
        for (size_t i = 0; i < modes; ++i) forces[n * modes + i] = source[i] * amplitude;
        diagnostics.push_back(Step(p, state, std::span(forces).subspan(n * modes, modes)));
        samples[2 * n] = Pickup(p, state, pickup1);
        samples[2 * n + 1] = Pickup(p, state, pickup2);
        std::copy(state.Q.begin(), state.Q.end(), states.begin() + n * modes);
    }
    const auto trace = gpu.Run(forces, pickup, steps, true);
    Check(prefix + "metal_pickup_short", Relative(std::span(trace.Samples).first(256), std::span(samples).first(256)), 1e-10);
    Check(prefix + "metal_modal_trajectory", Relative(trace.States, states), 1e-10);
    Check(prefix + "metal_pickup_longer", Relative(trace.Samples, samples), 1e-10);
    Check(prefix + "metal_final_state", Relative(gpu.GetState().Q, state.Q), 1e-10);
    const auto saved = gpu.GetState();
    gpu.EvaluatePotential(q);
    const auto retained = gpu.GetState();
    Check(prefix + "metal_potential_preserves_q", Relative(retained.Q, saved.Q), 0);
    Check(prefix + "metal_potential_preserves_previous", Relative(retained.Previous, saved.Previous), 0);
    Check(prefix + "metal_potential_preserves_psi", std::abs(retained.Psi - saved.Psi), 0);
    // An omitted force batch must ignore data left by the preceding strike.
    gpu.Reset(saved);
    const auto implicitQuiet = gpu.Run({}, pickup, 33, true);
    gpu.Reset(saved);
    const auto explicitQuiet = gpu.Run(std::vector<double>(33 * modes, 0.), pickup, 33, true);
    Check(prefix + "metal_empty_force_after_strike", Relative(implicitQuiet.States, explicitQuiet.States), 0);
    double quietPower = 0;
    for (const auto &d : implicitQuiet.Metrics) quietPower += std::abs(d.InputPower);
    Check(prefix + "metal_empty_force_input_power", quietPower, 0);
    gpu.Reset(InitialState(p));
    std::vector<double> chunked;
    for (size_t offset = 0; offset < steps; offset += 256) {
        auto part = gpu.Run(std::span(forces).subspan(offset * modes, 256 * modes), pickup, 256, true);
        chunked.insert(chunked.end(), part.Samples.begin(), part.Samples.end());
    }
    Check(prefix + "metal_reset_and_chunk_equivalence", Relative(chunked, trace.Samples), 0);
    double error = 0, scale = 0;
    for (size_t i = 0; i < steps; ++i) {
        error = std::max(error, std::abs(trace.Metrics[i].Energy - diagnostics[i].Energy));
        scale = std::max(scale, std::abs(diagnostics[i].Energy));
    }
    Check(prefix + "metal_energy_history", error / scale, 1e-10);
    gpu.Reset(InitialState(p));
    const auto zeros = gpu.Run({}, pickup, 16, true);
    Check(prefix + "metal_zero_state_is_stationary", Norm(zeros.States), 0);
    c.Nonlinear = false;
    const auto linear = Prepare(c);
    Gpu linear_gpu(linear, backend);
    auto linear_state = InitialState(linear, .02);
    linear_gpu.Reset(linear_state);
    const auto linear_trace = linear_gpu.Run({}, pickup, 256, true);
    std::vector<double> linear_energy(256), gpu_energy(256);
    for (size_t i = 0; i < 256; ++i) {
        linear_energy[i] = Step(linear, linear_state).Energy;
        gpu_energy[i] = linear_trace.Metrics[i].Energy;
    }
    Check(prefix + "metal_linear_energy", Relative(gpu_energy, linear_energy), 1e-10);
    linear_gpu.Reset(InitialState(linear));
    Check(prefix + "metal_linear_silence_energy", linear_gpu.Run({}, pickup, 1).Metrics.front().Energy, 0);
    c.Nonlinear = true;
    c.UseExact = false;
    c.NormType = 2;
    const auto norm2 = Prepare(c);
    Gpu norm2_gpu(norm2, backend);
    for (const bool moving : {false, true}) {
        auto initial = InitialState(norm2, .03);
        if (moving) initial.Previous[0] -= .0001;
        initial.Psi += .001;
        norm2_gpu.Reset(initial);
        auto expected = initial;
        const auto expected_metrics = Step(norm2, expected);
        const auto trace2 = norm2_gpu.Run({}, pickup, 1, true);
        const std::string suffix = moving ? "moving" : "stationary";
        Check(prefix + "metal_norm2_drift_" + suffix, Relative(norm2_gpu.GetState().Q, expected.Q), 1e-10);
        Check(prefix + "metal_norm2_energy_" + suffix, std::abs(trace2.Metrics[0].Energy / expected_metrics.Energy - 1), 1e-10);
    }
    norm2_gpu.Reset(InitialState(norm2));
    norm2_gpu.Run({}, pickup, 16);
    Check(prefix + "metal_norm2_silence", Norm(norm2_gpu.GetState().Q), 0);
}

void FftGridChecks() {
    auto boundary = Small();
    boundary.SampleRate = 128000;
    boundary.CutoffHz = 50000;
    boundary.Kappa = 2 * boundary.CutoffHz / (std::numbers::pi * 800);
    boundary.Ratio = 400;
    const auto cutoff = Prepare(boundary);
    Check("upstream_cutoff_rectangle", std::abs(double(cutoff.Mx) - 400), 0);
    Check("upstream_cutoff_active", std::abs(double(cutoff.ActiveCount) - 399), 0);
    for (int test = 0; test < 6; ++test) {
        auto c = Small();
        c.Ratio = 1;
        c.Oversampling = std::array{1., 1.5, 1.75, 2.5}[std::min(test, 3)];
        if (test >= 4) {
            // Large collocation grids exercise device-memory scratch without
            // the ill-conditioned inverse biharmonic operator of a thin plate.
            c.Ratio = test == 4 ? 1. : 1.1;
            c.Oversampling = 64;
        }
        const auto p = Prepare(c);
        Gpu gpu(p, TransformBackend::Fft);
        std::vector<double> q(p.Mx * p.My);
        for (size_t i = 0; i < q.size(); ++i)
            if (p.Active[i]) q[i] = 1e-4 * std::sin(double(i) + .3);
        const auto expected = EvaluatePotential(p, q), actual = gpu.EvaluatePotential(q);
        const auto name = "fft_grid_" + std::to_string(p.Nx) + "x" + std::to_string(p.Ny);
        Check(name + "_potential", std::abs(actual.Value / expected.Value - 1), 1e-11);
        Check(name + "_gradient", Relative(actual.Gradient, expected.Gradient), 1e-11);
        Check(name + "_airy", Relative(actual.Airy, expected.Airy), 1e-11);
    }
}

void ReferenceChecks(const std::string &filename, bool cpu_only, TransformBackend backend) {
    const auto data = json::parse(std::ifstream(filename));
    const auto &settings = data.contains("resolved_config") ? data.at("resolved_config") : data.at("config");
    Config c;
    c.SampleRate = settings.value("sample_rate", 44100.);
    c.CutoffHz = settings.value("cutoff_hz", 17000.);
    c.Kappa = settings.value("kappa", 60.);
    c.Ratio = settings.value("ratio", 1.1);
    c.Sigma0 = settings.value("sigma0", 1.3);
    c.Sigma1 = settings.value("sigma1", 1e-4);
    c.UseExact = settings.value("use_exact", true);
    c.NormType = settings.value("norm_type", 1);
    c.Oversampling = settings.value("oversampling", 1.5);
    c.Lambda = settings.value("lambda", 1000.);
    c.Epsilon = settings.value("epsilon", 1e-12);
    const auto p = Prepare(c);
    const auto &op = data.at("operator");
    if (op.is_null()) return;
    std::vector<double> q(p.Mx * p.My), gradient(q.size());
    const auto active = data.at("active_indices").get<std::vector<size_t>>();
    if (data.at("mx") != p.Mx || data.at("my") != p.My)
        throw std::runtime_error("Reference transform dimensions mismatch");
    if (active.size() != p.ActiveCount) throw std::runtime_error("Reference mode count mismatch");
    for (size_t i = 0; i < active.size(); ++i) {
        if (active[i] >= q.size() || !p.Active[active[i]]) throw std::runtime_error("Reference mode indexing mismatch");
        q[active[i]] = op.at("q_active").at(i);
        gradient[active[i]] = op.at("gradient_active").at(i);
    }
    const double value = op.at("potential");
    const auto cpu = EvaluatePotential(p, q);
    Check("upstream_cpu_potential", std::abs(cpu.Value / value - 1), 2e-12);
    Check("upstream_cpu_gradient", Relative(cpu.Gradient, gradient), 2e-12);
    if (!cpu_only) {
        Gpu gpu(p, backend);
        const auto metal = gpu.EvaluatePotential(q);
        Check("upstream_metal_potential", std::abs(metal.Value / value - 1), 1e-11);
        Check("upstream_metal_gradient", Relative(metal.Gradient, gradient), 1e-11);
    }
}
} // namespace

int main(int argc, char *const *argv) {
    bool cpu_only = false, fft = false;
    std::string output, reference;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--cpu") cpu_only = true;
        else if (argument == "--fft") fft = true;
        else if (argument == "--gate") {
        } else if (argument == "--output" && i + 1 < argc) output = argv[++i];
        else if (argument == "--reference" && i + 1 < argc) reference = argv[++i];
        else {
            std::cerr << "Usage: PlateTest [--cpu | --fft] [--gate] [--output report.json] [--reference upstream.json]\n";
            return 2;
        }
    }
    try {
        if (reference.empty()) {
            CpuChecks();
            if (!cpu_only) {
                GpuChecks();
                GpuChecks(TransformBackend::Fft);
                FftGridChecks();
            }
        } else ReferenceChecks(reference, cpu_only, fft ? TransformBackend::Fft : TransformBackend::Dense);
    } catch (const std::exception &error) {
        std::cerr << "PlateTest: " << error.what() << '\n';
        Results["exception"] = error.what();
        ++Failures;
    }
    Results["failures"] = Failures;
    if (!output.empty() && !(std::ofstream(output) << Results.dump(2) << '\n')) return 2;
    return Failures ? 1 : 0;
}
