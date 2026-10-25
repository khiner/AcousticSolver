#include "ElasticRun.h"
#include <chrono>

namespace elastic {
double StateScale(const Model &model) {
    double scale = std::sqrt(2 * model.Energy(model.Initial));
    for (const auto &s : model.Sources) {
        const auto &e = model.Elements[s.ElementIndex];
        scale += std::abs(s.Amplitude) * std::sqrt(s.Load.dot(e.EnergyMass * s.Load)) / s.Frequency;
    }
    return scale > 0 ? scale : 1;
}
namespace {
class CpuStepper final : public Stepper {
    const Model &M;
    Trace &T;
    Vec Q, Residual;

public:
    CpuStepper(const Model &model, Trace &trace) : M(model), T(trace), Q(model.Initial), Residual(Vec::Zero(model.Size)) {}
    void Advance(int frame, int frames, int substeps, double rate) override {
        const double dt = 1 / (rate * substeps);
        for (int f = frame; f < frame + frames; ++f) {
            for (int sub = 0; sub < substeps; ++sub) {
                Residual.setZero();
                for (int stage = 0; stage < 5; ++stage) {
                    const double time = f / rate + (sub + RkC[stage]) * dt;
                    const Vec rhs = M.Rhs(Q, time);
                    Residual = RkA[stage] * Residual + dt * rhs;
                    Q += RkB[stage] * Residual;
                }
                if (!Q.allFinite()) throw std::runtime_error("Nonfinite integration state; reduce dt");
            }
            for (int ch = 0; ch < int(M.Receivers.size()); ++ch) {
                const auto &r = M.Receivers[ch];
                T.Samples(f, ch) = r.Sign * r.Weights.dot(Q.segment(M.Elements[r.ElementIndex].Offset + r.Field * M.Nodes, M.Nodes));
            }
            T.Energy.row(f) = M.Diagnostics(Q, (f + 1) / rate).transpose();
        }
    }
    void Finish(Trace &trace) override { trace.State = Q; }
};
} // namespace
Trace Model::Run(double seconds, double sampleRate, const RunOptions &options) const {
    if (!std::isfinite(seconds) || seconds <= 0 || !std::isfinite(sampleRate) || sampleRate < 1 || !std::isfinite(options.Dt) || options.Dt < 0) throw std::invalid_argument("Invalid render duration, rate or timestep");
    const double frames = std::floor(seconds * sampleRate);
    if (frames < 1 || frames * (Receivers.size() + 5) >= 32000000) throw std::invalid_argument("Output storage exceeds limit");
    const double target = options.Dt > 0 ? options.Dt : DtLimit;
    const double count = std::max(1., std::ceil(1 / (sampleRate * target)));
    if (count > 1000000) throw std::invalid_argument("Too many substeps");
    const int substeps = int(count);
    Trace trace;
    trace.Samples.resize(int(frames), Receivers.size());
    trace.Energy.resize(int(frames), 5);
    trace.Dt = 1 / (sampleRate * substeps);
    trace.Steps = uint64_t(frames) * substeps;
    const auto start = std::chrono::steady_clock::now();
    std::unique_ptr<Stepper> stepper = options.Cpu ? std::make_unique<CpuStepper>(*this, trace) : GpuStepper(*this, trace);
    const auto prepared = std::chrono::steady_clock::now();
    trace.SetupSeconds = std::chrono::duration<double>(prepared - start).count();
    for (int frame = 0; frame < int(frames);) {
        const int batch = std::min({64, std::max(1, CommandSteps / substeps), int(frames) - frame});
        stepper->Advance(frame, batch, substeps, sampleRate);
        frame += batch;
    }
    stepper->Finish(trace);
    trace.Seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - prepared).count();
    if (!trace.State.allFinite() || !trace.Samples.allFinite() || !trace.Energy.allFinite()) throw std::runtime_error("Nonfinite render output");
    return trace;
}
} // namespace elastic
