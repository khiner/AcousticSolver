#include "PlateGpu.h"
#include "PlateFft.h"
#include "PlateMetal.h"
#include "PlatePrecision.h"
#include "Profile.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace plate {
namespace {
enum Kernel {
    DerivativeX,
    DerivativeProject,
    AiryY,
    Summary,
    PhiX,
    PhiForce,
    ForceY,
    Step
};

const char *KernelNames[]{"PlateDerivativeX", "PlateDerivativeProject", "PlateAiryY", "PlateSummary", "PlatePhiX", "PlatePhiForce", "PlateForceY", "PlateStep"};

using metal::Upload;

bool Finite(PlateReal value) {
    return value.Exponent != 0x7fffffff;
}

double ReadReal(PlateReal value) {
    const double result = ToDouble(value);
    if (!std::isfinite(result)) throw std::runtime_error("Plate Metal output became nonfinite");
    return result;
}

void Download(std::vector<double> &target, const PlateReal *values, size_t count) {
    target.resize(count);
    for (size_t i = 0; i < count; ++i)
        target[i] = ReadReal(values[i]);
}

void UploadReals(GpuBuffer &buffer, const std::vector<Precise> &values) { Upload(buffer, Pack(values)); }

PlateParams Parameters(const Prepared &p) {
    const Precise dt = Precise(1) / p.Settings.SampleRate, k2 = Precise(p.Settings.Kappa) * p.Settings.Kappa, scale = sqrt(Precise(p.Nx) * p.Ny);
    return {
        .Mx = uint32_t(p.Mx), .My = uint32_t(p.My), .Nx = uint32_t(p.Nx), .Ny = uint32_t(p.Ny), .Modes = uint32_t(p.Mx * p.My), .AiryModes = uint32_t((p.Mx + 1) * (p.My + 1)), .Grid = uint32_t(p.Nx * p.Ny), .Nonlinear = uint32_t(p.Settings.Nonlinear), .NormType = uint32_t(p.Settings.NormType), .Dt = Pack(dt), .Kappa2 = Pack(k2), .Lambda = Pack(Precise(p.Settings.Lambda)), .Epsilon = Pack(Precise(p.Settings.Epsilon)), .Scale = Pack(scale), .Dt2 = Pack(dt * dt), .StepKappa2 = Pack(dt * dt * k2), .HalfInvDt = Pack(Precise(.5) / dt), .InvDt2 = Pack(1 / (dt * dt)), .InvScale = Pack(1 / scale)
    };
}

Dim3 Blocks(size_t n) { return {uint32_t((n + 127) / 128), 1, 1}; }
} // namespace

Gpu::Gpu(const Prepared &prepared, TransformBackend backend) : Tables(prepared), BaseParams(Parameters(Tables)) {
    const auto p = BaseParams;
    if (!p.Modes || !p.Grid || !p.AiryModes || Tables.Mx * Tables.My != p.Modes ||
        Tables.Nx * Tables.Ny != p.Grid || !Finite(p.Kappa2) ||
        !Finite(p.Lambda) || !Finite(p.Dt) || !Finite(p.Epsilon) ||
        !Finite(p.Scale) || !Finite(p.Dt2) || !Finite(p.StepKappa2) || !Finite(p.HalfInvDt) || !Finite(p.InvDt2) || !Finite(p.InvScale) ||
        !(p.Dt2 > 0) || !(p.Epsilon > 0))
        throw std::invalid_argument("Plate configuration exceeds Metal range");
    const PreciseTables exact(Tables, backend == TransformBackend::Dense);
    Upload(Active, Tables.Active);
    Upload(AiryActive, Tables.AiryActive);
    if (backend == TransformBackend::Dense) {
        UploadReals(Sx, exact.Sx);
        UploadReals(Sy, exact.Sy);
        UploadReals(Cx, exact.Cx);
        UploadReals(Cy, exact.Cy);
    }
    const auto &kx = exact.Kx, &ky = exact.Ky;
    // Differentiation is diagonal in modal space. Fold those fixed factors into
    // each one-dimensional synthesis matrix once, before packing its values.
    const auto weighted =
        [](GpuBuffer &u, GpuBuffer &phi, const std::vector<Precise> &s,
           const std::vector<Precise> &c, const std::vector<Precise> &k,
           size_t nodes, bool xAxis) {
            const size_t modes = k.size() - 1;
            std::vector<Precise> uv(4 * modes * nodes), pv(4 * (modes + 1) * nodes);
            for (size_t mode = 0; mode <= modes; ++mode)
                for (size_t node = 0; node < nodes; ++node) {
                    const Precise wave = k[mode], cosine = c[mode * nodes + node];
                    const size_t offset = 4 * (mode * nodes + node);
                    pv[offset + (xAxis ? 0 : 1)] = -wave * wave * cosine;
                    pv[offset + (xAxis ? 1 : 0)] = cosine;
                    if (mode) {
                        const Precise &sine = s[(mode - 1) * nodes + node];
                        pv[offset + 2] = wave * sine;
                        const size_t uoffset = 4 * ((mode - 1) * nodes + node);
                        uv[uoffset + (xAxis ? 0 : 1)] = -wave * wave * sine;
                        uv[uoffset + (xAxis ? 1 : 0)] = sine;
                        uv[uoffset + 2] = wave * cosine;
                    }
                }
            UploadReals(u, uv);
            UploadReals(phi, pv);
        };
    if (backend == TransformBackend::Dense) {
        weighted(Ux, Px, exact.Sx, exact.Cx, kx, Tables.Nx, true);
        weighted(Uy, Py, exact.Sy, exact.Cy, ky, Tables.Ny, false);
    } else {
        Bracket.Resize(size_t(2) * p.Grid * sizeof(PlateReal));
        PhiBracket.Resize(size_t(p.Grid) * sizeof(PlateReal));
    }
    std::vector<Precise> airy(2 * p.AiryModes), coeff(size_t(p.Modes) * 4);
    for (size_t i = 0; i <= Tables.Mx; ++i)
        for (size_t j = 0; j <= Tables.My; ++j) {
            const size_t index = i * (Tables.My + 1) + j;
            const Precise k2 = kx[i] * kx[i] + ky[j] * ky[j];
            airy[index * 2] = k2;
            // The constant mode and truncated interior modes remain zero.
            if (Tables.AiryActive[index])
                airy[index * 2 + 1] = -1 / (exact.Scale * k2 * k2);
        }
    UploadReals(AiryCoefficients, airy);
    for (size_t i = 0; i < p.Modes; ++i) {
        coeff[i * 4] = exact.Dt * exact.Dt * exact.Omega2[i];
        coeff[i * 4 + 1] = exact.Sigma[i];
        coeff[i * 4 + 2] =
            exact.Dt * exact.Sigma[i] / (1 + exact.Dt * exact.Sigma[i]);
        coeff[i * 4 + 3] = exact.Omega2[i];
    }
    UploadReals(Coefficients, coeff);
    StateBuffer.Resize(size_t(p.Modes) * 4 * sizeof(PlateReal));
    EvaluationState.Resize(StateBuffer.Capacity());
    DerivativeX.Resize(size_t(2) * p.Nx * p.My * 4 * sizeof(PlateReal));
    Derivatives.Resize(size_t(p.Grid) * 4 * sizeof(PlateReal));
    ProjectionX.Resize(size_t(2) * (p.Mx + 1) * p.Ny * sizeof(PlateReal));
    Airy.Resize(size_t(2) * p.AiryModes * sizeof(PlateReal));
    PhiX.Resize(size_t(p.Nx) * (p.My + 1) * 4 * sizeof(PlateReal));
    ForceX.Resize(size_t(p.Mx) * p.Ny * sizeof(PlateReal));
    Gradient.Resize(size_t(p.Modes) * sizeof(PlateReal));
    Coupling.Resize(size_t(p.Modes) * 4 * sizeof(PlateReal));
    Scalars.ResizeZeroed(10 * sizeof(PlateReal));
    Forces.ResizeZeroed(size_t(p.Modes) * sizeof(PlateReal));
    States.Resize(sizeof(PlateReal));
    const auto library = metal::Compile(metal::Read(ACOUSTIC_PLATE_MSL_DIR "/PlateReal.h") + "\n" + metal::Read(ACOUSTIC_PLATE_MSL_DIR "/PlateParams.h") + "\n" + metal::Read(ACOUSTIC_PLATE_MSL_DIR "/PlateKernels.metal") + (backend == TransformBackend::Fft ? "\n" + metal::Read(ACOUSTIC_PLATE_MSL_DIR "/PlateFft.metal") : ""));
    for (size_t i = 0; i < Pipelines.size(); ++i)
        if (backend == TransformBackend::Dense || i == Summary || i == Kernel::Step)
            Pipelines[i] = metal::Pipeline(library.get(), KernelNames[i]);
    if (backend == TransformBackend::Fft) FastTransforms = std::make_unique<Fft>(Tables, library.get());
    ResetInitial();
}

Gpu::~Gpu() { MetalContext::Get().Drain(); }

void Gpu::Reset(const State &state) {
    const size_t n = Tables.Mx * Tables.My;
    if (state.Q.size() != n || state.Previous.size() != n ||
        !std::isfinite(state.Psi))
        throw std::invalid_argument("Invalid plate GPU state");
    std::vector<PlateReal> values(n * 4), scalars(10);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(state.Q[i]) || !std::isfinite(state.Previous[i]) ||
            (!Tables.Active[i] && (state.Q[i] != 0 || state.Previous[i] != 0)))
            throw std::invalid_argument("Invalid plate GPU modal state");
        values[i * 4] = MakeReal(state.Q[i]);
        values[i * 4 + 1] = MakeReal(state.Q[i]) - MakeReal(state.Previous[i]);
    }
    scalars[0] = MakeReal(state.Psi);
    MetalContext::Get().Drain();
    Upload(StateBuffer, values);
    Upload(Scalars, scalars);
    StepIndex = state.StepIndex;
}

void Gpu::ResetInitial(double amplitude) {
    std::vector<double> q(Tables.Mx * Tables.My);
    q[0] = amplitude;
    Reset({q, q, 0, 0});
    Nonlinearity(StateBuffer, true);
    MetalContext::Get().Drain();
    auto *scalars = Scalars.As<PlateReal>();
    scalars[0] = PlateSqrt(PlateReal(2.f) * scalars[2] + BaseParams.Epsilon);
}

State Gpu::GetState() {
    const size_t n = Tables.Mx * Tables.My;
    const auto *values = StateBuffer.As<PlateReal>();
    const auto *scalars = Scalars.As<PlateReal>();
    State result{std::vector<double>(n), std::vector<double>(n), ReadReal(scalars[0]), StepIndex};
    for (size_t i = 0; i < n; ++i) {
        result.Q[i] = ReadReal(values[i * 4]);
        result.Previous[i] = ReadReal(values[i * 4] - values[i * 4 + 1]);
    }
    return result;
}

void Gpu::Dispatch(size_t kernel, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice> buffers, const PlateParams &params, size_t threadgroupBytes) {
    metal::Dispatch(Pipelines[kernel].get(), KernelNames[kernel], blocks, threads, buffers, params, threadgroupBytes);
}

void Gpu::Nonlinearity(const GpuBuffer &state, bool forceNonlinear) {
    auto p = BaseParams;
    if (forceNonlinear)
        p.Nonlinear = 1;
    const auto transform = [&](PlateFftStage stage, const GpuBuffer &input, GpuBuffer &output) {
        FastTransforms->Dispatch(stage, input, output, Derivatives, AiryCoefficients, AiryActive, p);
    };
    if (p.Nonlinear) {
        if (FastTransforms) {
            transform(FftDerivativeX, state, DerivativeX);
            transform(FftDerivativeY, DerivativeX, Bracket);
            transform(FftProjectX, Bracket, ProjectionX);
            transform(FftAiryY, ProjectionX, Airy);
        } else {
            Dispatch(Kernel::DerivativeX, Blocks(PlateTransformLanes * size_t(2) * p.Nx * p.My), {128}, {&state, &Ux, &DerivativeX}, p);
            Dispatch(DerivativeProject, {2 * p.Ny}, {256}, {&DerivativeX, &Uy, &Derivatives, &Cx, &ProjectionX}, p, (p.Nx * sizeof(PlateReal) + 15) & ~size_t(15));
            Dispatch(AiryY, Blocks(PlateTransformLanes * size_t(2) * p.AiryModes), {128}, {&ProjectionX, &Cy, &AiryCoefficients, &AiryActive, &Airy}, p);
        }
    }
    Dispatch(Summary, {1}, {256}, {&Airy, &AiryCoefficients, &state, &Scalars}, p);
    if (p.Nonlinear) {
        if (FastTransforms) {
            transform(FftPhiX, Airy, PhiX);
            transform(FftPhiY, PhiX, PhiBracket);
            transform(FftForceX, PhiBracket, ForceX);
        } else {
            Dispatch(Kernel::PhiX, Blocks(PlateTransformLanes * size_t(p.Nx) * (p.My + 1)), {128}, {&Airy, &Px, &PhiX}, p);
            Dispatch(PhiForce, {p.Ny}, {256}, {&PhiX, &Py, &Derivatives, &Sx, &ForceX}, p, (p.Nx * sizeof(PlateReal) + 15) & ~size_t(15));
        }
    }
}

void Gpu::FinishGradient(const PlateParams &p) {
    if (FastTransforms)
        FastTransforms->Dispatch(FftForceY, ForceX, Gradient, Derivatives, AiryCoefficients, Active, p);
    else
        Dispatch(ForceY, Blocks(PlateTransformLanes * size_t(p.Modes)), {128}, {&ForceX, &Sy, &Active, &Gradient}, p);
}

Potential Gpu::EvaluatePotential(std::span<const double> q) {
    const size_t n = Tables.Mx * Tables.My;
    if (q.size() != n)
        throw std::invalid_argument("Wrong plate potential state size");
    std::vector<PlateReal> state(n * 4);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(q[i]) || (!Tables.Active[i] && q[i] != 0))
            throw std::invalid_argument("Invalid plate potential state");
        state[i * 4] = MakeReal(q[i]);
    }
    auto &ctx = MetalContext::Get();
    ctx.Drain();
    Upload(EvaluationState, state);
    Nonlinearity(EvaluationState, true);
    auto p = BaseParams;
    p.Nonlinear = 1;
    FinishGradient(p);
    const auto *scalars = Scalars.As<PlateReal>();
    Potential result{ReadReal(scalars[2]), {}, {}};
    const auto *gradient = Gradient.As<PlateReal>();
    Download(result.Gradient, gradient, n);
    Download(result.Airy, Airy.As<PlateReal>(), p.AiryModes);
    return result;
}

GpuTrace Gpu::Run(std::span<const double> forces, std::span<const double> pickupWeights, uint32_t steps, bool recordStates) {
    std::vector<PlateReal> f, w;
    f.reserve(forces.size());
    w.reserve(pickupWeights.size());
    for (double const v : forces) {
        if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite force");
        f.push_back(MakeReal(v));
    }
    for (double const v : pickupWeights) {
        if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite pickup");
        w.push_back(MakeReal(v));
    }
    return RunPrepared(f, w, steps, recordStates);
}
GpuTrace Gpu::RunPrepared(std::span<const PlateReal> forces, std::span<const PlateReal> pickupWeights, uint32_t steps, bool recordStates) {
    const size_t n = Tables.Mx * Tables.My;
    if ((!forces.empty() && forces.size() != size_t(steps) * n) ||
        pickupWeights.size() % n ||
        size_t(steps) * n > std::numeric_limits<uint32_t>::max() ||
        size_t(steps) * 6 > std::numeric_limits<uint32_t>::max() ||
        pickupWeights.size() > std::numeric_limits<uint32_t>::max() ||
        size_t(steps) * (pickupWeights.size() / n) >
            std::numeric_limits<uint32_t>::max() ||
        StepIndex > std::numeric_limits<uint64_t>::max() - steps)
        throw std::invalid_argument("Invalid plate GPU batch dimensions");
    for (const auto values : {forces, pickupWeights})
        for (const auto value : values)
            if (!Finite(value))
                throw std::invalid_argument("Nonfinite plate GPU input");
    if (!steps)
        return {};
    auto p = BaseParams;
    p.Receivers = uint32_t(pickupWeights.size() / n);
    p.RecordStates = recordStates;
    p.HasForces = !forces.empty();
    auto &ctx = MetalContext::Get();
    ctx.Drain();
    {
        const profile::Scope scope{"plate/upload"};
        if (forces.empty())
            Forces.Resize(sizeof(PlateReal));
        else
            Upload(Forces, forces);
        Upload(Pickups, pickupWeights);
        Samples.Resize(std::max(size_t(1), size_t(steps) * p.Receivers) * sizeof(PlateReal));
        Records.Resize(size_t(steps) * 6 * sizeof(PlateReal));
        if (recordStates)
            States.Resize(size_t(steps) * n * sizeof(PlateReal));
    }
    {
        const profile::Scope scope{"plate/encode"};
        for (uint32_t step = 0; step < steps; ++step) {
            p.Sample = step;
            Nonlinearity(StateBuffer);
            FinishGradient(p);
            Dispatch(Kernel::Step, {1}, {256}, {&Gradient, &Active, &StateBuffer, &Coefficients, &Scalars, &Forces, &Coupling, &Pickups, &Samples, &Records, &States}, p);
            // Bound command-buffer size even for callers requesting a long batch.
            if ((step + 1) % 32 == 0)
                ctx.Flush();
        }
    }
    StepIndex += steps;
    GpuTrace result;
    const auto *samples = Samples.As<PlateReal>();
    const profile::Scope readback{"plate/readback"};
    Download(result.Samples, samples, size_t(steps) * p.Receivers);
    const auto *records = Records.As<PlateReal>();
    result.Metrics.reserve(steps);
    for (size_t step = 0; step < steps; ++step) {
        const auto *d = records + step * 6;
        result.Metrics.push_back({ReadReal(d[0]), ReadReal(d[1]), ReadReal(d[2]), ReadReal(d[3]), ReadReal(d[4]), ReadReal(d[5])});
    }
    if (recordStates) {
        const auto *states = States.As<PlateReal>();
        Download(result.States, states, size_t(steps) * n);
    }
    const auto *state = StateBuffer.As<PlateReal>();
    for (size_t i = 0; i < n * 4; ++i)
        if (!Finite(state[i]))
            throw std::runtime_error("Plate Metal modal state became nonfinite");
    const auto *scalar = Scalars.As<PlateReal>();
    if (!Finite(scalar[0]))
        throw std::runtime_error("Plate Metal auxiliary state became nonfinite");
    return result;
}

} // namespace plate
