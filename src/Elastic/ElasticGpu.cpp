#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "ElasticRun.h"
#include <Metal/Metal.hpp>
#include <cstring>
#include <fstream>
#include <future>
#include <sstream>
#include <string_view>

namespace elastic {
namespace {
using Buffer = NS::SharedPtr<MTL::Buffer>;
using Pipeline = NS::SharedPtr<MTL::ComputePipelineState>;
constexpr int RowLanes = 32;
enum Access : uint32_t {
    StateA = 1,
    StateB = 2,
    RkResidual = 4,
    DiagnosticRows = 8,
    DiagnosticGroups = 16,
    EnergyOutput = 32,
    ReceiverOutput = 64,
    PhysicalA = 128,
    FluxValues = 256,
    PhysicalB = 512,
    RhsValues = 1024
};
struct alignas(8) Pair {
    float Hi, Lo;
};
Pair Pack(double x) {
    const float hi = float(x);
    if (!std::isfinite(hi)) throw std::runtime_error("Elastic GPU coefficient exceeds floating-point range");
    return {hi, float(x - hi)};
}
double Unpack(Pair x) { return double(x.Hi) + x.Lo; }
struct GpuMatrix {
    Buffer Starts, Columns, Values;
};
struct FaceData {
    uint32_t Inputs, Values, Rows, Outputs;
};
struct ElementData {
    uint32_t Offset, Fields, Mass;
};

struct StepParams {
    uint32_t Drive, Stage;
    Pair A, B, Dt;
};
struct RecordParams {
    uint32_t Count, Index;
};
class Gpu final : public Stepper {
    const Model &M;
    double Scale, GpuSeconds{};
    std::vector<Mat> Inverse;
    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL4::CommandQueue> Queue;
    NS::SharedPtr<MTL4::Compiler> Compiler;
    NS::SharedPtr<MTL4::CommandAllocator> Allocator;
    NS::SharedPtr<MTL4::CommandBuffer> Command;
    NS::SharedPtr<MTL4::ArgumentTable> Arguments;
    NS::SharedPtr<MTL::ResidencySet> Residency;
    MTL4::ComputeCommandEncoder *Encoder{};
    Buffer Parameters;
    size_t ParameterOffset{};
    std::array<uint64_t, 23> BoundAddresses{};
    uint32_t PendingReads{}, PendingWrites{}, QAccess{StateA}, NextAccess{StateB}, PhysicalAccess{PhysicalA}, NextPhysicalAccess{PhysicalB};
    bool ResidencyDirty{};
    NS::SharedPtr<MTL::Library> Library;
    Pipeline Sample, Diagnostic, Reduce, FinishDiagnostic, FaceFlux, MassStep, MassRhs;
    GpuMatrix P, B, S, R, InverseOperator, MassTransform;
    Buffer FaceInfo, FaceInputs, FaceMatrices, ElementInfo, FaceMap, Weak, MassBasis, MassCoefficients, Physical, NextPhysical, FaceValues, Rhs;
    Buffer Q, Next, Residual, Drives, Partial, Reduced, Samples, Energy;

    Buffer Allocate(size_t bytes, const void *data = nullptr) {
        const size_t length = std::max(size_t(8), bytes);
        auto buffer = NS::TransferPtr(Device->newBuffer(length, MTL::ResourceStorageModeShared));
        if (!buffer) throw std::runtime_error("Elastic GPU allocation failed");
        if (data && bytes) std::memcpy(buffer->contents(), data, bytes);
        else std::memset(buffer->contents(), 0, length);
        Residency->addAllocation(buffer.get());
        ResidencyDirty = true;
        return buffer;
    }
    template<class T> Buffer Upload(const std::vector<T> &values) { return Allocate(values.size() * sizeof(T), values.data()); }
    Buffer Upload(const Vec &values) {
        std::vector<Pair> packed(values.size());
        for (int i = 0; i < values.size(); ++i) packed[i] = Pack(values[i]);
        return Upload(packed);
    }
    // Column indices address contiguous blocks of width coefficients, including explicit zeros within each block.
    GpuMatrix Upload(const Sparse &matrix, int width) {
        std::vector<uint32_t> starts(matrix.rows() + 1), columns;
        std::vector<Pair> values;
        for (int row = 0; row < matrix.rows(); ++row) {
            starts[row] = columns.size();
            int previous = -1;
            for (Sparse::InnerIterator it(matrix, row); it; ++it) {
                const int base = (it.col() / width) * width;
                if (base != previous) {
                    columns.push_back(base);
                    values.resize(values.size() + width, Pair{});
                    previous = base;
                }
                values[values.size() - width + it.col() - base] = Pack(it.value());
            }
        }
        starts.back() = columns.size();
        return {Upload(starts), Upload(columns), Upload(values)};
    }
    Pipeline Compile(const char *name, bool update = false) {
        NS::Error *error{};
        auto const function = NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init());
        function->setLibrary(Library.get());
        function->setName(NS::String::string(name, NS::UTF8StringEncoding));
        auto const specialized = NS::TransferPtr(MTL4::SpecializedFunctionDescriptor::alloc()->init());
        auto const constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
        constants->setConstantValue(&update, MTL::DataTypeBool, NS::UInteger(0));
        specialized->setFunctionDescriptor(function.get());
        specialized->setConstantValues(constants.get());
        auto const descriptor = NS::TransferPtr(MTL4::ComputePipelineDescriptor::alloc()->init());
        descriptor->setComputeFunctionDescriptor(std::string_view(name) == "massStep" ? static_cast<MTL4::FunctionDescriptor *>(specialized.get()) : function.get());
        descriptor->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        auto pipeline = NS::TransferPtr(Compiler->newComputePipelineState(descriptor.get(), nullptr, &error));
        if (!pipeline) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "Cannot compile elastic GPU kernel");
        if (pipeline->threadExecutionWidth() != 32 || pipeline->maxTotalThreadsPerThreadgroup() < 256) throw std::runtime_error("Elastic GPU requires 32-lane SIMD groups and 256-thread groups");
        return pipeline;
    }
    void Address(uint64_t address, int index) {
        if (BoundAddresses[index] == address) return;
        Arguments->setAddress(address, index);
        BoundAddresses[index] = address;
    }
    void Bind(const Buffer &buffer, int index) { Address(buffer->gpuAddress(), index); }
    void Bind(const GpuMatrix &matrix, int offset) {
        Bind(matrix.Starts, offset);
        Bind(matrix.Columns, offset + 1);
        Bind(matrix.Values, offset + 2);
    }
    template<class T> void Params(const T &value, int index) {
        if (ParameterOffset + sizeof(T) > Parameters->length()) throw std::runtime_error("Elastic command parameter storage exhausted");
        std::memcpy(static_cast<char *>(Parameters->contents()) + ParameterOffset, &value, sizeof(T));
        Address(Parameters->gpuAddress() + ParameterOffset, index);
        ParameterOffset += (sizeof(T) + 15) & ~size_t(15);
    }
    void Dispatch(const Pipeline &pipeline, MTL::Size groups, MTL::Size threads, uint32_t reads, uint32_t writes) {
        if ((reads & PendingWrites) || (writes & (PendingReads | PendingWrites))) {
            Encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
            PendingReads = PendingWrites = 0;
        }
        Encoder->setComputePipelineState(pipeline.get());
        Encoder->dispatchThreadgroups(groups, threads);
        PendingReads |= reads;
        PendingWrites |= writes;
    }
    void Dispatch(const Pipeline &pipeline, int rows, uint32_t reads, uint32_t writes) {
        Dispatch(pipeline, MTL::Size((rows * RowLanes + 127) / 128, 1, 1), MTL::Size(128, 1, 1), reads, writes);
    }
    void Begin() {
        if (ResidencyDirty) {
            Residency->commit();
            ResidencyDirty = false;
        }
        Allocator->reset();
        Command->beginCommandBuffer(Allocator.get());
        Encoder = Command->computeCommandEncoder();
        Encoder->setArgumentTable(Arguments.get());
        ParameterOffset = 0;
        BoundAddresses.fill(0);
        PendingReads = PendingWrites = 0;
    }
    void Complete() {
        Encoder->endEncoding();
        Command->endCommandBuffer();
        const MTL4::CommandBuffer *commands[]{Command.get()};
        struct Feedback {
            double Seconds;
            std::string Error;
        };
        auto const completion = std::make_shared<std::promise<Feedback>>();
        auto result = completion->get_future();
        auto const options = NS::TransferPtr(MTL4::CommitOptions::alloc()->init());
        options->addFeedbackHandler(MTL4::CommitFeedbackHandlerFunction([completion](MTL4::CommitFeedback *feedback) {
            const double start = feedback->GPUStartTime(), end = feedback->GPUEndTime();
            completion->set_value({start > 0 && end >= start ? end - start : 0, feedback->error() ? feedback->error()->localizedDescription()->utf8String() : ""});
        }));
        Queue->commit(commands, 1, options.get());
        const auto feedback = result.get();
        GpuSeconds += feedback.Seconds;
        if (!feedback.Error.empty()) throw std::runtime_error(feedback.Error);
    }
    template<class Block> GpuMatrix UploadBlocks(Block blockAt) {
        std::vector<uint32_t> starts(M.Size + 1), columns;
        std::vector<Pair> values;
        for (int i = 0; i < int(M.Elements.size()); ++i) {
            const int offset = M.Elements[i].Offset;
            const Mat block = blockAt(i);
            for (int row = 0; row < block.rows(); ++row) {
                starts[offset + row] = columns.size();
                for (int col = 0; col < block.cols(); col += M.Nodes) {
                    if (block.row(row).segment(col, M.Nodes).isZero(0)) continue;
                    columns.push_back(offset + col);
                    for (int k = 0; k < M.Nodes; ++k) values.push_back(Pack(block(row, col + k)));
                }
            }
        }
        starts.back() = columns.size();
        return {Upload(starts), Upload(columns), Upload(values)};
    }

    void PrepareOperator(const std::vector<Mat> &u) {
        std::vector<ElementData> elements;
        std::vector<uint32_t> faceCounts(M.Elements.size()), faceMap(4 * M.Size, UINT32_MAX);
        std::vector<Pair> weak, mass, interpolation;
        const int nq = M.MassInterpolation.rows();
        for (int q = 0; q < nq; ++q)
            for (int n = 0; n < M.Nodes; ++n) interpolation.push_back(Pack(M.MassInterpolation(q, n)));
        for (int i = 0; i < int(M.Elements.size()); ++i) {
            const auto &e = M.Elements[i];
            elements.push_back({uint32_t(e.Offset), uint32_t(e.Fields()), uint32_t(mass.size())});
            for (const auto &w : e.Weak)
                for (int r = 0; r < M.Nodes; ++r)
                    for (int c = 0; c < M.Nodes; ++c) weak.push_back(Pack(w(r, c)));
            if (M.WeightAdjusted)
                for (int q = 0; q < nq; ++q) {
                    const auto &h = e.QuadratureInverse[q];
                    mass.push_back(Pack(M.Weights[q] * h(0, 0)));
                    if (e.Fluid) mass.push_back(Pack(M.Weights[q] * h(3, 3)));
                    else
                        for (int r = 3; r < 9; ++r)
                            for (int c = 3; c < 9; ++c) mass.push_back(Pack(M.Weights[q] * h(r, c)));
                }
        }
        std::vector<FaceData> faces;
        std::vector<uint32_t> inputs;
        std::vector<Pair> values;
        for (const auto &f : M.Faces) {
            const uint32_t offset = inputs.size(), leftRows = M.Elements[f.Left].Fields() * f.TraceNodes[0].size();
            const uint32_t rows = f.Load.rows();
            // Same-type interfaces use one flux with opposite signs on the two elements.
            const uint32_t outputs = f.Right >= 0 && M.Elements[f.Left].Fluid == M.Elements[f.Right].Fluid ? leftRows : rows;
            faces.push_back({offset, uint32_t(values.size()), rows, outputs});
            for (int col = 0; col < rows; ++col)
                for (int row = 0; row < outputs; ++row) values.push_back(Pack(f.Load(row, col)));
            for (int side = 0; side < (f.Right < 0 ? 1 : 2); ++side) {
                const int ei = side ? f.Right : f.Left;
                const auto &e = M.Elements[ei];
                const auto &nodes = f.TraceNodes[side];
                const int slot = faceCounts[ei]++;
                if (slot >= 4) throw std::runtime_error("Invalid tetrahedral face count");
                for (int field = 0; field < e.Fields(); ++field)
                    for (int const node : nodes) {
                        const int physical = e.Offset + field * M.Nodes + node;
                        faceMap[4 * physical + slot] = inputs.size();
                        inputs.push_back(physical);
                    }
            }
        }
        if (std::any_of(faceCounts.begin(), faceCounts.end(), [](uint32_t n) { return n != 4; })) throw std::runtime_error("Incomplete tetrahedral faces");
        const size_t faceRows = inputs.size();
        FaceInfo = Upload(faces);
        FaceInputs = Upload(inputs);
        FaceMatrices = Upload(values);
        InverseOperator = UploadBlocks([&](int i) { return Inverse[i]; });
        MassTransform = UploadBlocks([&](int i) { return M.WeightAdjusted ? u[i] : Mat(Inverse[i].transpose()); });
        ElementInfo = Upload(elements);
        FaceMap = Upload(faceMap);
        Weak = Upload(weak);
        MassBasis = Upload(interpolation);
        MassCoefficients = Upload(mass);
        Physical = Upload(Vec(M.Initial / Scale));
        NextPhysical = Allocate(M.Size * sizeof(Pair));
        FaceValues = Allocate(faceRows * sizeof(Pair));
        Rhs = Allocate(M.Size * sizeof(Pair));
    }
    void Evaluate(const StepParams &p, bool update) {
        Bind(FaceInfo, 0);
        Bind(FaceInputs, 1);
        Bind(FaceMatrices, 2);
        Bind(Physical, 3);
        Bind(FaceValues, 4);
        Dispatch(FaceFlux, MTL::Size(M.Faces.size(), 1, 1), MTL::Size(128, 1, 1), PhysicalAccess, FluxValues);
        Bind(ElementInfo, 0);
        Bind(MassBasis, 1);
        Bind(MassCoefficients, 2);
        Bind(Weak, 3);
        Bind(MassTransform, 4);
        Bind(Q, 7);
        Bind(Next, 8);
        Bind(Residual, 9);
        Bind(S, 10);
        Bind(Drives, 13);
        Params(p, 14);
        Bind(Rhs, 15);
        Bind(FaceMap, 16);
        Bind(FaceValues, 17);
        Bind(InverseOperator, 18);
        Bind(Physical, 21);
        Bind(NextPhysical, 22);
        const uint32_t reads = PhysicalAccess | FluxValues | (update ? QAccess | (p.Stage ? RkResidual : 0) : 0);
        Dispatch(update ? MassStep : MassRhs, MTL::Size(M.Elements.size(), 1, 1), MTL::Size(128, 1, 1), reads, RhsValues | (update ? NextAccess | NextPhysicalAccess | RkResidual : 0));
    }

    void Record(uint32_t drive) {
        Bind(Rhs, 0);
        Bind(P, 3);
        Bind(B, 6);
        Bind(S, 9);
        Bind(Q, 12);
        Bind(Drives, 13);
        Bind(Partial, 14);
        Params(RecordParams{uint32_t(M.Size), drive}, 15);
        Dispatch(Diagnostic, M.Size, QAccess | RhsValues, DiagnosticRows);
    }

public:
    Gpu(const Model &model, const Trace &trace) : M(model), Scale(StateScale(model)) {
        if (model.BoundaryState || model.PrescribedLoad || std::any_of(model.Faces.begin(), model.Faces.end(), [](const Face &f) { return f.Right < 0 && f.Boundary == "prescribed"; })) throw std::invalid_argument("Native prescribed boundary callbacks require the CPU backend");
        auto const pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        Device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        if (!Device) throw std::runtime_error("No Metal device");
        if (!Device->supportsFamily(MTL::GPUFamilyMetal4)) throw std::runtime_error("Elastic GPU requires Metal 4 and macOS 26 or newer");
        NS::Error *setupError{};
        Queue = NS::TransferPtr(Device->newMTL4CommandQueue());
        auto const compilerDescriptor = NS::TransferPtr(MTL4::CompilerDescriptor::alloc()->init());
        Compiler = NS::TransferPtr(Device->newCompiler(compilerDescriptor.get(), &setupError));
        Allocator = NS::TransferPtr(Device->newCommandAllocator());
        Command = NS::TransferPtr(Device->newCommandBuffer());
        auto const argumentDescriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        argumentDescriptor->setMaxBufferBindCount(23);
        argumentDescriptor->setInitializeBindings(true);
        Arguments = NS::TransferPtr(Device->newArgumentTable(argumentDescriptor.get(), &setupError));
        auto const residencyDescriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        residencyDescriptor->setInitialCapacity(64);
        Residency = NS::TransferPtr(Device->newResidencySet(residencyDescriptor.get(), &setupError));
        if (!Queue || !Compiler || !Allocator || !Command || !Arguments || !Residency) throw std::runtime_error(setupError ? setupError->localizedDescription()->utf8String() : "Cannot create elastic Metal 4 resources");
        Queue->addResidencySet(Residency.get());
        Parameters = Allocate(65536);
        std::ifstream const file(ACOUSTIC_ELASTIC_MSL_DIR "/Elastic.metal");
        if (!file) throw std::runtime_error("Cannot read Elastic.metal");
        // Leave half the threadgroup memory available for shader-validation instrumentation.
        int massTile = 128;
        const auto scratchBytes = [&] { return (18 * M.Nodes + 9 * (massTile + 1)) * sizeof(Pair) + 4; };
        while (massTile > 1 && scratchBytes() > Device->maxThreadgroupMemoryLength() / 2) massTile /= 2;
        if (scratchBytes() > Device->maxThreadgroupMemoryLength() / 2) throw std::runtime_error("Insufficient threadgroup memory for elastic quadrature");
        std::stringstream source;
        source << "#define MASS_TILE " << massTile << "\n"
               << "#define NODES " << M.Nodes << "\n#define FACE_NODES " << (M.Degree + 1) * (M.Degree + 2) / 2 << "\n#define VOL_Q " << M.MassInterpolation.rows() << "\n#define WEIGHT_ADJUSTED " << M.WeightAdjusted << "\n"
               << file.rdbuf();
        auto const options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
        options->setFastMathEnabled(false);
        options->setLanguageVersion(MTL::LanguageVersion4_0);
        NS::Error *error{};
        Library = NS::TransferPtr(Device->newLibrary(NS::String::string(source.str().c_str(), NS::UTF8StringEncoding), options.get(), &error));
        if (!Library) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "Cannot compile Elastic.metal");
        FaceFlux = Compile("faceFlux");
        MassStep = Compile("massStep", true);
        MassRhs = Compile("massStep");
        Sample = Compile("sample");
        Diagnostic = Compile("diagnostic");
        Reduce = Compile("reduceDiagnostic");
        FinishDiagnostic = Compile("finishDiagnostic");
        Entries sources, receivers;
        std::vector<Mat> u;
        Vec initial(M.Size);
        for (const auto &e : M.Elements) {
            const int n = M.Nodes * e.Fields();
            const Eigen::LLT<Mat> factor(e.EnergyMass);
            if (factor.info() != Eigen::Success) throw std::runtime_error("Energy mass is not positive definite");
            u.emplace_back(factor.matrixU());
            Inverse.emplace_back(factor.matrixU().solve(Mat::Identity(n, n)));
            initial.segment(e.Offset, n) = u.back() * (M.Initial.segment(e.Offset, n) / Scale);
        }
        for (int i = 0; i < int(M.Sources.size()); ++i) {
            const auto &s = M.Sources[i];
            AppendBlock(sources, u[s.ElementIndex] * (s.Load * (s.Amplitude / Scale)), M.Elements[s.ElementIndex].Offset, i);
        }
        for (int i = 0; i < int(M.Receivers.size()); ++i) {
            const auto &r = M.Receivers[i];
            const auto &e = M.Elements[r.ElementIndex];
            Vec w = Vec::Zero(M.Nodes * e.Fields());
            w.segment(r.Field * M.Nodes, M.Nodes) = r.Sign * r.Weights;
            AppendBlock(receivers, w.transpose() * Inverse[r.ElementIndex], i, e.Offset);
        }
        PrepareOperator(u);
        B = UploadBlocks([&](int i) -> Mat {
            const auto &e = M.Elements[i];
            const int n = M.Nodes * e.Fields();
            return Inverse[i].transpose() * Mat(M.BoundaryPower.block(e.Offset, e.Offset, n, n)) * Inverse[i];
        });
        P = UploadBlocks([&](int i) -> Mat {
            const auto &e = M.Elements[i];
            const int n = M.Nodes * e.Fields();
            return e.EnergyMass == e.PhysicalMass ? Mat(Mat::Identity(n, n)) : Mat(Inverse[i].transpose() * e.PhysicalMass * Inverse[i]);
        });
        S = Upload(MakeSparse(M.Size, M.Sources.size(), sources), 1);
        R = Upload(MakeSparse(M.Receivers.size(), M.Size, receivers), M.Nodes);
        Q = Upload(initial);
        Next = Allocate(Q->length());
        Residual = Allocate(Q->length());
        Partial = Allocate(M.Size * 5 * sizeof(Pair));
        Reduced = Allocate(((M.Size + 1023) / 1024) * 5 * sizeof(Pair));
        Samples = Allocate(trace.Samples.size() * sizeof(Pair));
        Energy = Allocate(trace.Energy.size() * sizeof(Pair));
    }
    void Advance(int frame, int frames, int substeps, double rate) override {
        auto const pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        const double dt = 1 / (rate * substeps);
        const int ns = M.Sources.size();
        const size_t stageDrives = size_t(frames) * substeps * 5;
        if ((stageDrives + 1) * std::max(ns, 1) > 32000000) throw std::runtime_error("Source drive storage exceeds limit");
        std::vector<Pair> drives((stageDrives + 1) * ns);
        const auto drive = [&](size_t index, double time) {
            for (int i = 0; i < ns; ++i) drives[index * ns + i] = Pack(M.Sources[i].Drive(time));
        };
        for (int f = 0; f < frames; ++f)
            for (int sub = 0; sub < substeps; ++sub)
                for (int stage = 0; stage < 5; ++stage) drive((size_t(f) * substeps + sub) * 5 + stage, (frame + f) / rate + (sub + RkC[stage]) * dt);
        drive(stageDrives, (frame + frames) / rate);
        const size_t driveBytes = drives.size() * sizeof(Pair);
        if (!Drives || Drives->length() < driveBytes) {
            if (Drives) Residency->removeAllocation(Drives.get());
            Drives = Allocate(driveBytes);
        }
        if (driveBytes) std::memcpy(Drives->contents(), drives.data(), driveBytes);
        Begin();
        const auto RecordEnergy = [&](int outputFrame) {
            Bind(Partial, 0);
            Bind(Reduced, 1);
            Params(RecordParams{uint32_t(M.Size), uint32_t(outputFrame)}, 2);
            const uint32_t groups = (M.Size + 1023) / 1024;
            Dispatch(Reduce, MTL::Size(groups, 1, 1), MTL::Size(256, 1, 1), DiagnosticRows, DiagnosticGroups);

            Bind(Reduced, 0);
            Bind(Energy, 1);
            Params(RecordParams{groups, uint32_t(outputFrame)}, 2);
            Dispatch(FinishDiagnostic, MTL::Size(5, 1, 1), MTL::Size(32, 1, 1), DiagnosticGroups, EnergyOutput);
        };
        int encoded = 0;
        for (int f = 0; f < frames; ++f) {
            for (int sub = 0; sub < substeps; ++sub) {
                for (int stage = 0; stage < 5; ++stage) {
                    const StepParams p{uint32_t(((size_t(f) * substeps + sub) * 5 + stage) * ns), uint32_t(stage), Pack(RkA[stage]), Pack(RkB[stage]), Pack(dt)};
                    Evaluate(p, true);
                    if (f > 0 && sub == 0 && stage == 0) Record(p.Drive);
                    std::swap(Q, Next);
                    std::swap(QAccess, NextAccess);
                    std::swap(Physical, NextPhysical);
                    std::swap(PhysicalAccess, NextPhysicalAccess);
                    // Stage one and the previous sample's energy reduction access disjoint buffers and may overlap.
                    if (f > 0 && sub == 0 && stage == 1) RecordEnergy(frame + f - 1);
                }
                if (++encoded == CommandSteps) {
                    Complete();
                    encoded = 0;
                    Begin();
                }
            }
            if (!M.Receivers.empty()) {
                Bind(R, 0);
                Bind(Q, 3);
                Bind(Samples, 4);
                Params(RecordParams{uint32_t(M.Receivers.size()), uint32_t(frame + f)}, 5);
                Dispatch(Sample, M.Receivers.size(), QAccess, ReceiverOutput);
            }
            if (f == frames - 1) {
                const uint32_t drive = stageDrives * ns;
                Evaluate(StepParams{}, false);
                Record(drive);
                RecordEnergy(frame + f);
            }
        }
        Complete();
    }
    void Finish(Trace &trace) override {
        trace.GpuSeconds = GpuSeconds;
        const auto *q = static_cast<const Pair *>(Q->contents()), *samples = static_cast<const Pair *>(Samples->contents()), *energy = static_cast<const Pair *>(Energy->contents());
        Vec state(M.Size);
        for (int i = 0; i < M.Size; ++i) state[i] = Unpack(q[i]);
        trace.State.resize(M.Size);
        for (int i = 0; i < int(M.Elements.size()); ++i) {
            const auto &e = M.Elements[i];
            trace.State.segment(e.Offset, Inverse[i].rows()) = Scale * (Inverse[i] * state.segment(e.Offset, Inverse[i].rows()));
        }
        for (int f = 0; f < trace.Samples.rows(); ++f) {
            for (int ch = 0; ch < trace.Samples.cols(); ++ch) trace.Samples(f, ch) = Scale * Unpack(samples[f * trace.Samples.cols() + ch]);
            for (int col = 0; col < 5; ++col) trace.Energy(f, col) = Scale * Scale * Unpack(energy[f * 5 + col]);
        }
    }
};
} // namespace
std::unique_ptr<Stepper> GpuStepper(const Model &model, Trace const &trace) { return std::make_unique<Gpu>(model, trace); }
} // namespace elastic
