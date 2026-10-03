// Heightfield sampling follows Acoustic Reliefs acoustics_opt.py.
#include "BemTexture.h"
#include "BemParams.h"
#include "Metal/MetalContext.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bem {
namespace {
template<class T> void Upload(GpuBuffer &buffer, const T *data, size_t count) {
    buffer.Resize(std::max(count, size_t(1)) * sizeof(T));
    if (count) buffer.Upload(data, count * sizeof(T));
}
void Dimensions(int w, int h) {
    if (w < 2 || h < 2 || w > 4096 || h > 4096) throw std::invalid_argument("BEM texture dimensions must be 2..4096");
}
} // namespace
UVs HeightUVs(const Problem &p) {
    const double xmin = p.Vertices.col(0).minCoeff(), xmax = p.Vertices.col(0).maxCoeff();
    const double zmin = p.Vertices.col(2).minCoeff(), zmax = p.Vertices.col(2).maxCoeff();
    if (!(xmax > xmin && zmax > zmin)) throw std::invalid_argument("BEM heightfield needs nonzero x/z extent");
    UVs uv(p.HeightVertices.size(), 2);
    for (size_t i = 0; i < p.HeightVertices.size(); ++i) {
        uv(i, 0) = (p.Vertices(p.HeightVertices[i], 0) - xmin) / (xmax - xmin) * .9998 + .0001;
        uv(i, 1) = (p.Vertices(p.HeightVertices[i], 2) - zmin) / (zmax - zmin) * .9998 + .0001;
    }
    return uv;
}
struct HeightfieldMap::Impl {
    GpuBuffer Stencils, Offsets, Links, Pixels, Values, Gradient;
    BemTextureParams Params{};
    Impl(const UVs &uv, int w, int h) {
        Dimensions(w, h);
        if (uv.rows() < 1 || !uv.allFinite() || (uv.array() < 0.).any() || (uv.array() > 1.).any()) throw std::invalid_argument("Invalid BEM texture UVs");
        Params.Width = w;
        Params.Height = h;
        Params.Count = uv.rows();
        std::vector<BemTexStencil> stencils(uv.rows());
        std::vector<std::vector<BemTexLink>> adjacent(w * h);
        for (int i = 0; i < uv.rows(); ++i) {
            double ui, vi;
            double u = std::modf(uv(i, 0) * w - .5, &ui), v = std::modf(uv(i, 1) * h - .5, &vi);
            // Match np.modf plus index clamping, including its lower-edge behavior.
            if (u < 0.) u += 1.;
            if (v < 0.) v += 1.;
            const int x = std::max(int(ui), 0), y = std::max(int(vi), 0);
            const int x1 = std::min(x + 1, w - 1), y1 = std::min(y + 1, h - 1);
            const std::array<uint32_t, 4> indices{uint32_t(y * w + x), uint32_t(y * w + x1), uint32_t(y1 * w + x), uint32_t(y1 * w + x1)};
            const std::array<float, 4> weights{float((1. - v) * (1. - u)), float((1. - v) * u), float(v * (1. - u)), float(v * u)};
            stencils[i] = {indices[0], indices[1], indices[2], indices[3], {weights[0], weights[1], weights[2], weights[3]}};
            for (int j = 0; j < 4; ++j) adjacent[indices[j]].push_back({uint32_t(i), weights[j]});
        }
        std::vector<uint32_t> offsets{0};
        std::vector<BemTexLink> links;
        for (const auto &list : adjacent) {
            links.insert(links.end(), list.begin(), list.end());
            offsets.push_back(links.size());
        }
        Upload(Stencils, stencils.data(), stencils.size());
        Upload(Offsets, offsets.data(), offsets.size());
        Upload(Links, links.data(), links.size());
        Pixels.Resize(size_t(w) * h * sizeof(float));
        Values.Resize(uv.rows() * sizeof(float));
        Gradient.Resize(size_t(w) * h * sizeof(float));
    }
};
HeightfieldMap::HeightfieldMap(const UVs &uv, int w, int h) : Data(std::make_unique<Impl>(uv, w, h)) {}
HeightfieldMap::~HeightfieldMap() = default;
Eigen::VectorXd HeightfieldMap::Sample(const Texture &pixels) {
    auto &d = *Data;
    if (pixels.size() != d.Params.Width * d.Params.Height || !pixels.allFinite()) throw std::invalid_argument("Invalid BEM height texture");
    d.Pixels.Upload(pixels.data(), pixels.size() * sizeof(float));
    auto &ctx = MetalContext::Get();
    ctx.Dispatch(ctx.BemPipeline("bem_texture_sample"), {(d.Params.Count + 63) / 64, 1, 1}, {64, 1, 1}, {&d.Stencils, &d.Pixels, &d.Values}, &d.Params, sizeof(d.Params));
    return Eigen::Map<Eigen::VectorXf>(d.Values.As<float>(), d.Params.Count).cast<double>();
}
Texture HeightfieldMap::Transpose(const Eigen::VectorXd &gradient) {
    auto &d = *Data;
    if (gradient.size() != d.Params.Count || !gradient.allFinite()) throw std::invalid_argument("Invalid BEM height gradient");
    const Texture packed = gradient.cast<float>();
    d.Values.Upload(packed.data(), packed.size() * sizeof(float));
    auto &ctx = MetalContext::Get();
    const uint32_t count = d.Params.Width * d.Params.Height;
    ctx.Dispatch(ctx.BemPipeline("bem_texture_transpose"), {(count + 63) / 64, 1, 1}, {64, 1, 1}, {&d.Offsets, &d.Links, &d.Values, &d.Gradient}, &d.Params, sizeof(d.Params));
    return Eigen::Map<Texture>(d.Gradient.As<float>(), count);
}
} // namespace bem
