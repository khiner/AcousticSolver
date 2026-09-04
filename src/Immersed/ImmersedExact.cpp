#include "ImmersedExact.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace immersed {
namespace {
using Complex = std::complex<double>;

double Norm(const Point &p) {
    double result = 0.;
    for (double const x : p) result += x * x;
    return std::sqrt(result);
}

double Dot(const Point &a, const Point &b) {
    double result = 0.;
    for (int i = 0; i < 3; ++i) result += a[size_t(i)] * b[size_t(i)];
    return result;
}

std::vector<double> SphericalJ(double x, int order) {
    if (!(x > 0.)) throw std::runtime_error("Spherical Bessel argument must be positive");
    const int start = order + 50 + int(x);
    std::vector<double> work(size_t(start) + 2, 0.);
    work[size_t(start)] = 1.;
    for (int l = start; l > 0; --l) {
        work[size_t(l) - 1] = (2. * l + 1.) / x * work[size_t(l)] - work[size_t(l) + 1];
        if (std::abs(work[size_t(l) - 1]) > 1e150) {
            for (int j = l - 1; j <= start + 1; ++j) work[size_t(j)] *= 1e-150;
        }
    }
    const double scale = (std::sin(x) / x) / work[0];
    std::vector<double> result(size_t(order) + 1);
    for (int l = 0; l <= order; ++l) result[size_t(l)] = work[size_t(l)] * scale;
    return result;
}

std::vector<double> SphericalY(double x, int order) {
    std::vector<double> result(size_t(order) + 1);
    result[0] = -std::cos(x) / x;
    if (order == 0) return result;
    result[1] = -std::cos(x) / (x * x) - std::sin(x) / x;
    for (int l = 1; l < order; ++l)
        result[size_t(l) + 1] = (2. * l + 1.) / x * result[size_t(l)] - result[size_t(l) - 1];
    return result;
}

struct SphericalSet {
    std::vector<double> J, Jp;
    std::vector<Complex> H, Hp;
};

Complex BoundaryRatio(const SphericalSet &values, int order, double alpha) {
    if (std::isinf(alpha)) return values.Jp[size_t(order)] / values.Hp[size_t(order)];
    return (values.J[size_t(order)] + Complex{0., alpha} * values.Jp[size_t(order)]) /
        (values.H[size_t(order)] + Complex{0., alpha} * values.Hp[size_t(order)]);
}

SphericalSet Spherical(double x, int order) {
    SphericalSet result;
    result.J = SphericalJ(x, order + 1);
    const auto y = SphericalY(x, order + 1);
    result.Jp.resize(size_t(order) + 1);
    result.H.resize(size_t(order) + 1);
    result.Hp.resize(size_t(order) + 1);
    for (int l = 0; l <= order; ++l)
        result.H[size_t(l)] = {result.J[size_t(l)], -y[size_t(l)]};
    for (int l = 0; l <= order; ++l) {
        result.Jp[size_t(l)] = l == 0 ? -result.J[1] : result.J[size_t(l - 1)] - (l + 1.) / x * result.J[size_t(l)];
        result.Hp[size_t(l)] = l == 0 ? -result.H[1] : result.H[size_t(l - 1)] - (l + 1.) / x * result.H[size_t(l)];
    }
    result.J.resize(size_t(order) + 1);
    return result;
}

int PickOrder(double largest_argument, int requested) {
    if (requested >= 0) return requested;
    return std::max(100, int(std::ceil(largest_argument)) + 24);
}

std::vector<double> Legendre(double x, int order) {
    std::vector<double> result(size_t(order) + 1);
    result[0] = 1.;
    if (order == 0) return result;
    result[1] = x;
    for (int l = 1; l < order; ++l)
        result[size_t(l) + 1] = ((2. * l + 1.) * x * result[size_t(l)] - l * result[size_t(l) - 1]) / (l + 1.);
    return result;
}

} // namespace

std::complex<double> FreeMonopoleTransfer(double angular_frequency, double sound_speed, double density, double distance) {
    if (!(sound_speed > 0.) || !(density > 0.) || !(distance > 0.))
        throw std::runtime_error("Free monopole parameters must be positive");
    if (angular_frequency == 0.) return {};
    if (angular_frequency < 0.)
        return std::conj(FreeMonopoleTransfer(-angular_frequency, sound_speed, density, distance));
    const double k = angular_frequency / sound_speed;
    return Complex{0., angular_frequency * density} * std::exp(Complex{0., -k * distance}) /
        (4. * std::numbers::pi * distance);
}

std::complex<double> FreeSphereSeriesTransfer(double angular_frequency, double sound_speed, double density, const Point &source, const Point &receiver, int max_order) {
    if (angular_frequency == 0.) return {};
    if (angular_frequency < 0.)
        return std::conj(FreeSphereSeriesTransfer(-angular_frequency, sound_speed, density, source, receiver, max_order));
    const double rs = Norm(source), ro = Norm(receiver);
    if (!(rs > 0.) || !(ro > 0.)) throw std::runtime_error("Sphere-series points cannot be at the origin");
    const double k = angular_frequency / sound_speed;
    const int order = PickOrder(k * std::max(rs, ro), max_order);
    const auto inner = Spherical(k * std::min(rs, ro), order);
    const auto outer = Spherical(k * std::max(rs, ro), order);
    const auto legendre = Legendre(std::clamp(Dot(source, receiver) / (rs * ro), -1., 1.), order);
    Complex sum{};
    for (int l = 0; l <= order; ++l)
        sum += (2. * l + 1.) * legendre[size_t(l)] * inner.J[size_t(l)] * outer.H[size_t(l)];
    return angular_frequency * density * k * sum / (4. * std::numbers::pi);
}

std::complex<double> SphereTransfer(double angular_frequency, double sound_speed, double density, double radius, const Point &source, const Point &receiver, double alpha, int max_order) {
    if (angular_frequency == 0.) return {};
    if (angular_frequency < 0.)
        return std::conj(SphereTransfer(-angular_frequency, sound_speed, density, radius, source, receiver, alpha, max_order));
    const double rs = Norm(source), ro = Norm(receiver);
    if (!(radius > 0.) || ro < radius || rs <= radius || ro > rs)
        throw std::runtime_error("Sphere transfer requires R <= receiver radius <= source radius");
    const double k = angular_frequency / sound_speed;
    const int order = PickOrder(k * rs, max_order);
    const auto at_radius = Spherical(k * radius, order);
    const auto at_source = Spherical(k * rs, order);
    const auto at_receiver = Spherical(k * ro, order);
    const auto legendre = Legendre(std::clamp(Dot(source, receiver) / (rs * ro), -1., 1.), order);
    Complex sum{};
    for (int l = 0; l <= order; ++l) {
        const Complex ratio = BoundaryRatio(at_radius, l, alpha);
        const Complex radial = at_receiver.J[size_t(l)] - at_receiver.H[size_t(l)] * ratio;
        sum += (2. * l + 1.) * legendre[size_t(l)] * at_source.H[size_t(l)] * radial;
    }
    return angular_frequency * density * k * sum / (4. * std::numbers::pi);
}

double SphereBoundaryResidual(double angular_frequency, double sound_speed, double radius, double source_radius, double alpha, bool derivative, int max_order) {
    const double k = std::abs(angular_frequency) / sound_speed;
    const int order = max_order >= 0 ? max_order : std::max(20, int(std::ceil(k * source_radius)) + 24);
    const auto at_radius = Spherical(k * radius, order);
    const auto &j = derivative ? at_radius.Jp : at_radius.J;
    const auto &h = derivative ? at_radius.Hp : at_radius.H;
    double worst = 0.;
    for (int l = 0; l <= order; ++l) {
        const Complex ratio = BoundaryRatio(at_radius, l, alpha);
        const Complex value = j[size_t(l)] - ratio * h[size_t(l)];
        const double scale = std::max(std::abs(j[size_t(l)]), std::abs(ratio * h[size_t(l)]));
        if (scale > std::numeric_limits<double>::min()) worst = std::max(worst, std::abs(value) / scale);
    }
    return worst;
}

} // namespace immersed
