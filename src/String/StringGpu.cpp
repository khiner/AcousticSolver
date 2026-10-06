#include "StringGpu.h"
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace stiffstring {
namespace {
std::string ReadShader(const char *name) {
    std::ifstream input(std::string(ACOUSTIC_STRING_MSL_DIR) + "/" + name);
    if (!input) throw std::runtime_error(std::string("Cannot read string shader: ") + name);
    return {std::istreambuf_iterator<char>(input), {}};
}
StringReal Pack(double value) {
    const float hi = float(value);
    StringReal packed{hi, float(value - double(hi))};
    if (!std::isfinite(value) || !std::isfinite(packed.Hi) || (value && packed.Hi == 0))
        throw std::invalid_argument("String value exceeds Metal float-pair range");
    return packed;
}
double Read(StringReal value) {
    double const result = double(value.Hi) + value.Lo;
    if (!std::isfinite(result)) throw std::runtime_error("String Metal output became nonfinite");
    return result;
}
void Download(std::vector<double> &output, const GpuBuffer &buffer, size_t count) {
    output.resize(count);
    const auto *data = buffer.As<StringReal>();
    std::transform(data, data + count, output.begin(), Read);
}
} // namespace

Gpu::Gpu(const Prepared &p) : Tables(p) {
    if (!p.Points || p.Points > std::numeric_limits<uint32_t>::max() / 2 || (p.Components != 1 && p.Components != 2))
        throw std::invalid_argument("Invalid string GPU dimensions");
    if (p.Receiver[0] >= p.Points || (p.Components == 2 && p.Receiver[1] >= p.Points))
        throw std::invalid_argument("Invalid string GPU receiver indices");
    Params = {
        .Points = uint32_t(p.Points), .Components = uint32_t(p.Components), .Model = uint32_t(p.Settings.Equation), .Split = uint32_t(p.Settings.Split), .Nonlinear = uint32_t(p.Settings.Nonlinear), .Reference = uint32_t(p.Settings.Method == Integrator::Reference), .Steps = 0, .Offset = 0, .RecordStates = 0, .ReceiverU = uint32_t(p.Receiver[0]), .ReceiverV = uint32_t(p.Receiver[1]), .H = Pack(p.H), .InvH = Pack(1 / p.H), .InvH2 = Pack(1 / (p.H * p.H)), .Dt = Pack(p.Dt), .Dt2 = Pack(p.Dt * p.Dt), .InvDt = Pack(1 / p.Dt), .Mass = Pack(p.Mass), .Rigidity = Pack(p.Rigidity), .EA = Pack(p.EA), .Tension = Pack(p.Settings.Tension), .LongitudinalTension = Pack(p.LongitudinalTension), .NonlinearModulus = Pack(p.NonlinearModulus), .Sigma0 = Pack(p.Sigma0), .Sigma1 = Pack(p.Sigma1), .InverseMass = Pack(p.InverseMass), .Shift = Pack(p.Settings.Shift), .Length = Pack(p.Settings.Length)
    };
    auto const options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
    options->setFastMathEnabled(false);
    std::string const source = ReadShader("StringReal.h") + "\n" + ReadShader("StringParams.h") + "\n" + ReadShader("StringKernels.metal");
    NS::Error *error{};
    auto const library = NS::TransferPtr(MetalContext::Get().Device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), options.get(), &error));
    if (!library) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "String shader compilation failed");
    auto const function = NS::TransferPtr(library->newFunction(NS::String::string("StringAdvance", NS::UTF8StringEncoding)));
    if (!function) throw std::runtime_error("Missing StringAdvance Metal kernel");
    Pipeline = NS::TransferPtr(MetalContext::Get().Device->newComputePipelineState(function.get(), &error));
    if (!Pipeline) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "String pipeline creation failed");
    if (Pipeline->maxTotalThreadsPerThreadgroup() < 256)
        throw std::runtime_error("String Metal requires 256-thread groups");
    const size_t dimension = p.Points * p.Components;
    StateBuffer.Resize(5 * dimension * sizeof(StringReal));
    const size_t tridiagonal = Params.Reference && p.Settings.Equation == Model::Cubic ? 8 * p.Points : 0;
    Scratch.Resize((2 * (p.Points + 1) + 3 * dimension + tridiagonal) * sizeof(StringReal));
    Scalars.Resize(2 * sizeof(StringReal));
    States.Resize(sizeof(StringReal));
    Reset(InitialState(Tables));
}
Gpu::~Gpu() { MetalContext::Get().Drain(); }
void Gpu::Reset(const State &state) {
    const size_t dimension = Tables.Points * Tables.Components;
    if (state.Q.size() != dimension || state.Previous.size() != dimension)
        throw std::invalid_argument("Invalid string GPU state dimensions");
    std::vector<StringReal> values(5 * dimension);
    std::transform(state.Q.begin(), state.Q.end(), values.begin(), Pack);
    std::transform(state.Previous.begin(), state.Previous.end(), values.begin() + dimension, Pack);
    for (size_t i = 0; i < dimension; ++i) values[3 * dimension + i] = Pack(state.Q[i] - state.Previous[i]);
    StringReal scalars[]{Pack(state.Psi), Pack(state.Loss)};
    MetalContext::Get().Drain();
    StateBuffer.Upload(values.data(), values.size() * sizeof(StringReal));
    Scalars.Upload(scalars, sizeof(scalars));
    StepIndex = state.StepIndex;
}
State Gpu::GetState() {
    const size_t dimension = Tables.Points * Tables.Components;
    const auto *values = StateBuffer.As<StringReal>();
    State result{.Q = std::vector<double>(dimension), .Previous = std::vector<double>(dimension)};
    std::transform(values, values + dimension, result.Q.begin(), Read);
    std::transform(values + dimension, values + 2 * dimension, result.Previous.begin(), Read);
    const auto *scalars = Scalars.As<StringReal>();
    result.Psi = Read(scalars[0]);
    result.Loss = Read(scalars[1]);
    result.StepIndex = StepIndex;
    return result;
}
GpuTrace Gpu::Run(uint32_t steps, bool recordStates) {
    GpuTrace trace;
    if (!steps) return trace;
    const size_t dimension = Tables.Points * Tables.Components;
    if (StepIndex > std::numeric_limits<uint64_t>::max() - steps)
        throw std::overflow_error("String step counter overflow");
    if (size_t(steps) > std::numeric_limits<size_t>::max() / sizeof(StringReal) / std::max<size_t>(6, recordStates ? dimension : 0))
        throw std::overflow_error("String trace allocation overflow");
    Samples.Resize(size_t(steps) * Tables.Components * sizeof(StringReal));
    Records.Resize(size_t(steps) * 6 * sizeof(StringReal));
    if (recordStates) States.Resize(size_t(steps) * dimension * sizeof(StringReal));
    auto &context = MetalContext::Get();
    const uint32_t batchSteps = uint32_t(std::clamp<size_t>(8192 / dimension, 1, 32));
    size_t queuedWork = 0;
    for (uint32_t offset = 0; offset < steps;) {
        auto params = Params;
        params.Steps = std::min(batchSteps, steps - offset);
        params.Offset = offset;
        params.RecordStates = recordStates;
        context.Dispatch(Pipeline.get(), {1, 1, 1}, {256, 1, 1}, {&StateBuffer, &Scratch, &Scalars, &Samples, &Records, &States}, &params, sizeof(params));
        offset += params.Steps;
        // Bound command buffers as well as individual dispatches for offline runs.
        queuedWork += size_t(params.Steps) * dimension;
        if (queuedWork >= 1048576) {
            context.Drain();
            queuedWork = 0;
        }
    }
    context.Drain();
    StepIndex += steps;
    Download(trace.Samples, Samples, size_t(steps) * Tables.Components);
    if (recordStates) Download(trace.States, States, size_t(steps) * dimension);
    const auto *values = Records.As<StringReal>();
    trace.Metrics.resize(steps);
    for (size_t i = 0; i < steps; ++i)
        trace.Metrics[i] = {.Kinetic = Read(values[6 * i]), .LinearPotential = Read(values[6 * i + 1]), .NonlinearPotential = Read(values[6 * i + 2]), .Dissipated = Read(values[6 * i + 3]), .Energy = Read(values[6 * i + 4]), .Psi = Read(values[6 * i + 5])};
    return trace;
}
} // namespace stiffstring
