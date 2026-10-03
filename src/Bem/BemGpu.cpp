#include "BemGpu.h"
#include "BemParams.h"
#include "Metal/MetalContext.h"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace bem {
using namespace detail;
Eigen::VectorXd MetalGradient(const Problem &p, const Vector &x, const Vector &vf, const Vector &lambda) {
    const Geometry geometry(p);
    auto params = geometry.Params;
    params.PointCount = p.Listeners.rows();
    GpuBuffer offsetsBuffer, cornersBuffer, contributions, surface, sensitivity, adjoint, incidentDy, gradient;
    std::vector<int> heightIndex(p.Vertices.rows(), -1);
    for (size_t i = 0; i < p.HeightVertices.size(); ++i) heightIndex[p.HeightVertices[i]] = int(i);
    std::vector<std::vector<uint32_t>> adjacent(p.HeightVertices.size());
    for (int t = 0; t < p.Faces.rows(); ++t)
        for (int c = 0; c < 3; ++c) {
            const int index = heightIndex[p.Faces(t, c)];
            if (index >= 0) adjacent[index].push_back(3 * t + c);
        }
    std::vector<uint32_t> offsets{0}, corners;
    for (const auto &list : adjacent) {
        corners.insert(corners.end(), list.begin(), list.end());
        offsets.push_back(corners.size());
    }
    offsetsBuffer.Resize(offsets.size() * sizeof(uint32_t));
    offsetsBuffer.Upload(offsets.data(), offsets.size() * sizeof(uint32_t));
    cornersBuffer.Resize(std::max(size_t(1), corners.size()) * sizeof(uint32_t));
    if (!corners.empty()) cornersBuffer.Upload(corners.data(), corners.size() * sizeof(uint32_t));
    contributions.Resize(3 * p.Faces.rows() * sizeof(Complex));
    UploadVector(surface, x);
    UploadVector(sensitivity, vf);
    UploadVector(adjoint, lambda);
    UploadVector(incidentDy, SourceValues(p, true));
    gradient.Resize(p.HeightVertices.size() * sizeof(float));
    auto &ctx = MetalContext::Get();
    ctx.Dispatch(ctx.BemPipeline("bem_gradient_faces"), {uint32_t(corners.size()), 1, 1}, {128, 1, 1}, {&geometry.Faces, &geometry.Listeners, &surface, &sensitivity, &adjoint, &incidentDy, &contributions, &geometry.ListenerPhases, &cornersBuffer}, &params, sizeof(params));
    ctx.Dispatch(ctx.BemPipeline("bem_gradient_gather"), {(params.HeightCount + 63) / 64, 1, 1}, {64, 1, 1}, {&offsetsBuffer, &cornersBuffer, &contributions, &gradient}, &params, sizeof(params));
    const auto *data = gradient.As<float>();
    Eigen::VectorXd result(p.HeightVertices.size());
    for (size_t i = 0; i < p.HeightVertices.size(); ++i) result[i] = data[i];
    return result;
}
} // namespace bem
