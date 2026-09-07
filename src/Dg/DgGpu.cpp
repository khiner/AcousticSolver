#include "DgGpu.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace dg {
namespace {
template<typename T> void Upload(GpuBuffer &buffer, const std::vector<T> &values) {
    if (values.empty()) throw std::runtime_error("Empty DG input buffer");
    buffer.Resize(values.size() * sizeof(T));
    buffer.Upload(values.data(), values.size() * sizeof(T));
}

std::string Read(const char *path) {
    std::ifstream const input(path);
    if (!input) throw std::runtime_error("Cannot read DG shader source");
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

std::vector<float> Download(const GpuBuffer &buffer, size_t count) {
    const auto *data = buffer.As<float>();
    return {data, data + count};
}
} // namespace

Gpu::Gpu(const Tables &tables) : Params(tables.Params), RkA(tables.RkA), RkB(tables.RkB) {
    const size_t ecount = Params.Elements, n = Params.Nodes, f = Params.FaceNodes, k = f / 4;
    // Strong and weak volume terms share three matrices through transposition.
    std::vector<float> volume(ecount * n * n * 3);
    double volume_max = 0, transpose_error = 0;
    for (size_t e = 0; e < ecount; ++e)
        for (size_t axis = 0; axis < 3; ++axis)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) {
                    const float value = tables.G[((e * 6 + axis) * n + i) * n + j];
                    volume_max = std::max(volume_max, double(std::abs(value)));
                    transpose_error = std::max(transpose_error, std::abs(double(value) - tables.G[((e * 6 + axis + 3) * n + j) * n + i]));
                    volume[((e * 3 + axis) * n + i) * n + j] = value;
                }
    if (transpose_error > 1e-12 * volume_max) throw std::runtime_error("Inconsistent weak volume transpose");
    Upload(G, volume);
    if (f % 4) throw std::runtime_error("Four tetrahedron faces required");
    std::vector<int32_t> rows(4 * n, -1);
    for (size_t face = 0; face < 4; ++face)
        for (size_t row = 0; row < k; ++row) rows[4 * (tables.VmapM[face * k + row] % n) + face] = int32_t(row);
    for (size_t e = 0; e < ecount; ++e)
        for (size_t j = 0; j < f; ++j)
            if (tables.VmapM[e * f + j] != e * n + tables.VmapM[j]) throw std::runtime_error("Inconsistent reference face ordering");
    std::vector<float> faces(ecount * 4 * 10 * k * k);
    double max_value = 0, discarded = 0;
    for (size_t e = 0; e < ecount; ++e)
        for (size_t term = 0; term < 10; ++term)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < f; ++j) {
                    const float value = tables.L[((e * 10 + term) * n + i) * f + j];
                    max_value = std::max(max_value, double(std::abs(value)));
                    const int32_t row = rows[i * 4 + j / k];
                    if (row < 0) discarded = std::max(discarded, double(std::abs(value)));
                    else faces[(((e * 4 + j / k) * 10 + term) * k + row) * k + j % k] = value;
                }
    // A volume test basis vanishes on every face that does not contain its node.
    if (discarded > 1e-12 * max_value) throw std::runtime_error("Nonzero weak load outside a face trace");
    // Each coefficient is an integral of two face basis functions times a scalar.
    const size_t triangle = k * (k + 1) / 2;
    std::vector<float> symmetric(ecount * 4 * 10 * triangle);
    double face_symmetry_error = 0;
    for (size_t block = 0; block < ecount * 4 * 10; ++block)
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j <= i; ++j) {
                const float value = faces[(block * k + i) * k + j];
                symmetric[block * triangle + i * (i + 1) / 2 + j] = value;
                face_symmetry_error = std::max(face_symmetry_error, std::abs(double(value) - faces[(block * k + j) * k + i]));
            }
    if (face_symmetry_error > 1e-12 * max_value) throw std::runtime_error("Nonsymmetric weak face load");
    // A planar face uses one scalar mass matrix and a constant unit normal.
    // Retain all ten matrices unless their reconstruction agrees to FP32 precision.
    std::vector<float> packed_faces;
    std::vector<DgFace> face_info;
    for (size_t face = 0; face < ecount * 4; ++face) {
        const float *a = symmetric.data() + face * 10 * triangle;
        size_t pivot = 0;
        for (size_t j = 1; j < triangle; ++j)
            if (std::abs(a[j]) > std::abs(a[pivot])) pivot = j;
        double nx = a[triangle + pivot] / double(a[pivot]), ny = a[2 * triangle + pivot] / double(a[pivot]), nz = a[3 * triangle + pivot] / double(a[pivot]);
        const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (!(norm > 0) || !std::isfinite(norm)) throw std::runtime_error("Invalid DG face normal");
        nx /= norm;
        ny /= norm;
        nz /= norm;
        const std::array<double, 10> coefficients{1, nx, ny, nz, nx * nx, nx * ny, nx * nz, ny * ny, ny * nz, nz * nz};
        double error = 0, magnitude = 0;
        for (size_t term = 0; term < 10; ++term)
            for (size_t j = 0; j < triangle; ++j) {
                error = std::max(error, std::abs(a[term * triangle + j] - coefficients[term] * a[j]));
                magnitude = std::max(magnitude, double(std::abs(a[term * triangle + j])));
            }
        const bool planar = error <= 1e-7 * magnitude;
        if (packed_faces.size() >= 0x80000000u) throw std::runtime_error("DG face storage exceeds offset range");
        face_info.push_back({float(nx), float(ny), float(nz), uint32_t(packed_faces.size()) | (planar ? 0x80000000u : 0u)});
        packed_faces.insert(packed_faces.end(), a, a + (planar ? 1 : 10) * triangle);
    }
    Upload(L, packed_faces);
    Upload(FaceInfo, face_info);
    Upload(FaceRows, rows);
    // Geometry is fixed: materialize T^T diag(W_e) T once, retaining the WADG mass approximation.
    const size_t quadrature = Params.QuadraturePoints;
    std::vector<double> transform(tables.T.begin(), tables.T.end()), weighted(transform.size()), matrix(n * n);
    std::vector<float> inverse_mass(ecount * n * n);
    for (size_t e = 0; e < ecount; ++e) {
        for (size_t q = 0; q < quadrature; ++q)
            for (size_t i = 0; i < n; ++i) weighted[q * n + i] = double(tables.W[e * quadrature + q]) * transform[q * n + i];
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, int(n), int(n), int(quadrature), 1., transform.data(), int(n), weighted.data(), int(n), 0., matrix.data(), int(n));
        for (size_t i = 0; i < n * n; ++i) inverse_mass[e * n * n + i] = float(matrix[i]);
    }
    Upload(InverseMass, inverse_mass);
    Upload(Vm, tables.VmapM);
    Upload(Vp, tables.VmapP);
    Upload(Boundary, tables.Boundary);
    Upload(ReceiverElements, tables.ReceiverElements);
    Upload(ReceiverWeights, tables.ReceiverWeights);
    const size_t state_bytes = size_t(Params.Elements) * Params.Nodes * 4 * sizeof(float);
    Q.Resize(state_bytes);
    Residual.Resize(state_bytes);
    Load.Resize(state_bytes);
    Rhs.Resize(state_bytes);
    auto &ctx = MetalContext::Get();
    if (!ctx.Device->supportsFamily(MTL::GPUFamilyApple7)) throw std::runtime_error("DG requires Apple GPU family 7 or newer");
    auto *options = MTL::CompileOptions::alloc()->init();
    options->setFastMathEnabled(false);
    const auto source = Read(ACOUSTIC_DG_MSL_DIR "/DgParams.h") + Read(ACOUSTIC_DG_MSL_DIR "/DgKernels.metal");
    NS::Error *error{};
    Library = ctx.Device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), options, &error);
    options->release();
    if (!Library) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "DG shader compilation failed");
    auto *constants = MTL::FunctionConstantValues::alloc()->init();
    constants->setConstantValue(&Params.Nodes, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&Params.FaceNodes, MTL::DataTypeUInt, NS::UInteger(2));
    auto const pipeline = [&](const char *name) {
        auto *function = Library->newFunction(NS::String::string(name, NS::UTF8StringEncoding), constants, &error);
        if (!function) throw std::runtime_error("Missing DG kernel");
        auto *result = ctx.Device->newComputePipelineState(function, &error);
        function->release();
        if (!result) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "DG pipeline creation failed");
        return result;
    };
    Weak = pipeline("DgWeakLoad");
    Project = pipeline("DgMassProject");
    constants->release();
    if (Project->threadExecutionWidth() != 32 || Weak->threadExecutionWidth() != 32)
        throw std::runtime_error("DG kernels require 32-lane SIMD groups");
    if (Weak->maxTotalThreadsPerThreadgroup() < 32 * DG_NODE_SIMDS || Project->maxTotalThreadsPerThreadgroup() < 32 * DG_NODE_SIMDS)
        throw std::runtime_error("DG threadgroup size exceeds pipeline limits");
}

Gpu::~Gpu() {
    MetalContext::Get().Drain();
    Weak->release();
    Project->release();
    Library->release();
}

void Gpu::Reset(std::span<const float> initial, uint32_t steps) {
    if (initial.size() != size_t(Params.Elements) * Params.Nodes * 4 || !steps) throw std::runtime_error("Invalid DG state or step count");
    MetalContext::Get().Drain();
    Params.Steps = steps;
    Record.Resize(size_t(Params.Receivers) * steps * sizeof(float));
    Q.Upload(initial.data(), initial.size_bytes());
    Residual.Zero(initial.size_bytes());
}

void Gpu::WeakLoad() {
    const uint32_t nodes_per_group = DG_NODE_SIMDS * 32 / DgNodeLanes(Params.Nodes);
    MetalContext::Get().Dispatch(Weak, {(Params.Elements * Params.Nodes + nodes_per_group - 1) / nodes_per_group}, {32 * DG_NODE_SIMDS}, {&Q, &G, &L, &Vm, &Vp, &Boundary, &FaceRows, &Load, &ReceiverElements, &ReceiverWeights, &Record, &FaceInfo}, &Params, sizeof(Params));
}

void Gpu::Mass() {
    const uint32_t nodes_per_group = DG_NODE_SIMDS * 32 / DgNodeLanes(Params.Nodes);
    MetalContext::Get().Dispatch(Project, {(Params.Elements * Params.Nodes + nodes_per_group - 1) / nodes_per_group}, {32 * DG_NODE_SIMDS}, {&Load, &InverseMass, &Rhs, &Q, &Residual}, &Params, sizeof(Params));
}

void Gpu::Step(uint32_t sample, float dt) {
    if (sample >= Params.Steps || !(dt > 0)) throw std::runtime_error("Invalid DG sample or timestep");
    Params.Sample = sample;
    Params.Dt = dt;
    Params.Update = 1;
    for (size_t stage = 0; stage < RkA.size(); ++stage) {
        Params.RkA = RkA[stage];
        Params.RkB = RkB[stage];
        WeakLoad();
        Mass();
    }
}

std::vector<float> Gpu::State() { return Download(Q, size_t(Params.Elements) * Params.Nodes * 4); }
std::vector<float> Gpu::Receivers() { return Download(Record, size_t(Params.Receivers) * Params.Steps); }

std::vector<float> Gpu::ApplyRhs(std::span<const float> state) {
    Reset(state, 1);
    Params.Update = 0;
    WeakLoad();
    Mass();
    return Download(Rhs, state.size());
}

std::vector<float> Gpu::ApplyMass(std::span<const float> load) {
    if (load.size() != size_t(Params.Elements) * Params.Nodes * 4) throw std::runtime_error("Invalid DG mass load");
    MetalContext::Get().Drain();
    Load.Upload(load.data(), load.size_bytes());
    Params.Update = 0;
    Mass();
    return Download(Rhs, load.size());
}

} // namespace dg
