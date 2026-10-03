#pragma once
#include "Bem.h"
#include "BemParams.h"
#include "Metal/MetalContext.h"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace bem::detail {
struct Complex {
    float Real, Imag;
};
static_assert(sizeof(Complex) == 8);
inline BemFloat4 Pack(const Eigen::Vector3d &v) {
    BemFloat4 packed{float(v.x()), float(v.y()), float(v.z()), 0.f};
    if (!std::isfinite(packed.x) || !std::isfinite(packed.y) || !std::isfinite(packed.z)) throw std::invalid_argument("BEM coordinates exceed Metal precision range");
    return packed;
}
inline void UploadVector(GpuBuffer &buffer, const Vector &v) {
    std::vector<Complex> packed(v.size());
    for (int i = 0; i < v.size(); ++i) packed[i] = {float(v[i].real()), float(v[i].imag())};
    buffer.Resize(packed.size() * sizeof(Complex));
    buffer.Upload(packed.data(), packed.size() * sizeof(Complex));
}
struct Geometry {
    GpuBuffer Faces, Listeners, ListenerPhases;
    BemParams Params{};
    explicit Geometry(const Problem &p) {
        const Eigen::Vector3d origin = p.Vertices.colwise().mean();
        std::vector<BemFace> faces(p.Faces.rows());
        for (int i = 0; i < p.Faces.rows(); ++i) {
            const Eigen::Vector3d a = p.Vertices.row(p.Faces(i, 0)).transpose() - origin;
            const auto high = Pack(a);
            const Eigen::Vector3d low = a - Eigen::Vector3d{high.x, high.y, high.z};
            faces[i] = {high, Pack((p.Vertices.row(p.Faces(i, 1)) - p.Vertices.row(p.Faces(i, 0))).transpose()), Pack((p.Vertices.row(p.Faces(i, 2)) - p.Vertices.row(p.Faces(i, 0))).transpose()), Pack(low)};
        }
        Faces.Resize(faces.size() * sizeof(BemFace));
        Faces.Upload(faces.data(), faces.size() * sizeof(BemFace));
        std::vector<BemFloat4> listeners(p.Listeners.rows());
        for (int i = 0; i < p.Listeners.rows(); ++i) listeners[i] = Pack(p.Listeners.row(i).transpose() - origin);
        Listeners.Resize(listeners.size() * sizeof(BemFloat4));
        Listeners.Upload(listeners.data(), listeners.size() * sizeof(BemFloat4));
        Vector phases(p.Listeners.rows());
        const double k = 2. * std::numbers::pi * p.Frequency / p.SoundSpeed;
        for (int i = 0; i < p.Listeners.rows(); ++i) phases[i] = std::exp(std::complex<double>{0., -k * (p.Listeners.row(i).transpose() - origin).norm()});
        UploadVector(ListenerPhases, phases);
        Params.FaceCount = p.Faces.rows();
        Params.HeightCount = p.HeightVertices.size();
        Params.Wavenumber = float(2. * std::numbers::pi * p.Frequency / p.SoundSpeed);
        if (!std::isfinite(Params.Wavenumber)) throw std::invalid_argument("BEM wavenumber exceeds Metal precision range");
    }
};
// Keep the distant-source phase in FP64: k*r can be much larger than the mesh's
// acoustic phase, so rounding it to float introduces avoidable incident-wave error.
inline Vector SourceValues(const Problem &p, bool derivative) {
    const double k = 2. * std::numbers::pi * p.Frequency / p.SoundSpeed;
    Vector values(p.Faces.rows());
    for (int i = 0; i < p.Faces.rows(); ++i) {
        const Eigen::Vector3d center = (p.Vertices.row(p.Faces(i, 0)) + p.Vertices.row(p.Faces(i, 1)) + p.Vertices.row(p.Faces(i, 2))) / 3.;
        const Eigen::Vector3d r = center - p.Source;
        const double d = r.norm();
        values[i] = std::exp(std::complex<double>{0., -k * d}) / (4. * std::numbers::pi * d);
        if (derivative) values[i] *= std::complex<double>{-1. / d, -k} * r.y() / (3. * d);
    }
    return values;
}
} // namespace bem::detail
