// Independent fixed-grid MPFR convergence reference.
#pragma once
// MPFR dense transforms are an accuracy reference, not a production backend.
#include "Plate/Plate.h"
#include "json.hpp"
#include <Eigen/Dense>
#include <boost/multiprecision/eigen.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

using Json = nlohmann::json;
template<class R> using Matrix = Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template<class R> using Vec = std::vector<R>;
using std::abs;
using std::acos;
using std::cos;
using std::exp;
using std::sin;
using std::sqrt;
using std::tanh;

template<class R> struct State {
    Vec<R> q, previous;
    R psi;
};
template<class R> struct Potential {
    R value{};
    Vec<R> gradient;
};
template<class R> struct Model {
    const plate::Prepared &p;
    R dt, kappa, epsilon, scale;
    Vec<R> kx, ky, ax, ay, omega, sigma, d;
    Matrix<R> sx, sy, cx, cy;
    explicit Model(const plate::Prepared &prepared) : p(prepared), dt(R(1) / R(p.Settings.SampleRate)), kappa(p.Settings.Kappa), epsilon(p.Settings.Epsilon), scale(sqrt(R(p.Nx) * R(p.Ny))) {
        const R pi = acos(R(-1)), ratio = p.Settings.Ratio;
        const R root = sqrt(ratio);
        auto table = [&](size_t modes, size_t nodes, bool cosine) {
            Matrix<R> t(modes + (cosine ? 1 : 0), nodes);
            for (size_t i = 0; i < size_t(t.rows()); ++i)
                for (size_t j = 0; j < nodes; ++j) {
                    R phase = pi * R(i + (cosine ? 0 : 1)) * (R(j) + R(.5)) / R(nodes);
                    t(i, j) = R(sqrt(R(cosine && i == 0 ? 1 : 2) / R(nodes)) * (cosine ? R(cos(phase)) : R(sin(phase))));
                }
            return t;
        };
        sx = table(p.Mx, p.Nx, false);
        sy = table(p.My, p.Ny, false);
        cx = table(p.Mx, p.Nx, true);
        cy = table(p.My, p.Ny, true);
        for (size_t x = 0; x <= p.Mx; ++x) ax.push_back(R(R(x) * pi / root));
        for (size_t y = 0; y <= p.My; ++y) ay.push_back(R(R(y) * pi * root));
        for (size_t x = 0; x < p.Mx; ++x)
            for (size_t y = 0; y < p.My; ++y) {
                auto i = x * p.My + y;
                kx.push_back(ax[x + 1]);
                ky.push_back(ay[y + 1]);
                R k2 = kx[i] * kx[i] + ky[i] * ky[i], w = kappa * k2, s = R(p.Settings.Sigma0) + R(p.Settings.Sigma1) * k2;
                R e = exp(-s * dt), e2 = exp(-2 * s * dt), phase = sqrt(w * w - s * s) * dt;
                omega.push_back(2 / (dt * dt) * (1 - 2 * e * cos(phase) + e2) / (1 + e2));
                sigma.push_back((1 - e2) / (dt * (1 + e2)));
                d.push_back(1 + dt * sigma.back());
            }
    }
    Matrix<R> synthesize(const Matrix<R> &q, bool cosine) const {
        const auto &tx = cosine ? cx : sx;
        const auto &ty = cosine ? cy : sy;
        const Matrix<R> tmp = tx.transpose() * q;
        return scale * (tmp * ty);
    }
    Matrix<R> analyze(const Matrix<R> &q, bool cosine) const {
        const auto &tx = cosine ? cx : sx;
        const auto &ty = cosine ? cy : sy;
        const Matrix<R> tmp = tx * q;
        return (tmp * ty.transpose()) / scale;
    }
    Potential<R> potential(const Vec<R> &q, bool gradient = true) const {
        Matrix<R> xx(p.Mx, p.My), yy(p.Mx, p.My), xy = Matrix<R>::Zero(p.Mx + 1, p.My + 1);
        for (size_t x = 0; x < p.Mx; ++x)
            for (size_t y = 0; y < p.My; ++y) {
                auto i = x * p.My + y;
                xx(x, y) = -kx[i] * kx[i] * q[i];
                yy(x, y) = -ky[i] * ky[i] * q[i];
                xy(x + 1, y + 1) = kx[i] * ky[i] * q[i];
            }
        const Matrix<R> uxx = synthesize(xx, false), uyy = synthesize(yy, false), uxy = synthesize(xy, true);
        Matrix<R> airy = analyze((2 * (uxx.array() * uyy.array() - uxy.array().square())).matrix(), true);
        Matrix<R> px(p.Mx + 1, p.My + 1), py(p.Mx + 1, p.My + 1), pxy(p.Mx, p.My);
        Potential<R> out;
        for (size_t x = 0; x <= p.Mx; ++x)
            for (size_t y = 0; y <= p.My; ++y) {
                R k2 = ax[x] * ax[x] + ay[y] * ay[y];
                R xi = p.AiryActive[x * (p.My + 1) + y] ? R(-airy(x, y) / (k2 * k2)) : R(0);
                airy(x, y) = xi;
                out.value += R(.25) * (k2 * xi) * (k2 * xi);
                px(x, y) = -ax[x] * ax[x] * xi;
                py(x, y) = -ay[y] * ay[y] * xi;
                if (x && y) pxy(x - 1, y - 1) = ax[x] * ay[y] * xi;
            }
        if (!gradient) return out;
        const Matrix<R> phix = synthesize(px, true), phiy = synthesize(py, true), phixy = synthesize(pxy, false);
        const Matrix<R> grad = analyze((uxx.array() * phiy.array() + uyy.array() * phix.array() - 2 * uxy.array() * phixy.array()).matrix(), false);
        out.gradient.resize(q.size());
        for (size_t i = 0; i < q.size(); ++i) out.gradient[i] = p.Active[i] ? R(-grad.data()[i]) : R(0);
        return out;
    }
    Vec<R> g(const Potential<R> &op) const {
        Vec<R> v = op.gradient;
        R denom = sqrt(2 * op.value + epsilon);
        for (auto &x : v) x /= denom;
        return v;
    }
    void advance(State<R> &s, const Vec<R> &g, const Vec<R> &force = {}) const {
        const R dt2 = dt * dt, nl = dt2 * kappa * kappa, beta = R(.25) * nl;
        Vec<R> delta(s.q.size()), r(s.q.size());
        R gd = 0, gr = 0, gh = 0;
        for (size_t i = 0; i < s.q.size(); ++i) {
            delta[i] = s.q[i] - s.previous[i];
            gd += g[i] * delta[i];
        }
        for (size_t i = 0; i < s.q.size(); ++i)
            if (p.Active[i]) {
                R rhs = (1 - dt * sigma[i]) * delta[i] - dt2 * omega[i] * s.q[i] - nl * g[i] * s.psi - beta * g[i] * gd;
                if (!force.empty()) rhs += dt2 * force[i];
                r[i] = rhs / d[i];
                gr += g[i] * r[i];
                gh += g[i] * g[i] / d[i];
            }
        R correction = beta * gr / (1 + beta * gh), psiIncrement = 0;
        Vec<R> next(s.q.size());
        for (size_t i = 0; i < s.q.size(); ++i) {
            R inc = r[i] - g[i] / d[i] * correction;
            next[i] = s.q[i] + inc;
            psiIncrement += R(.5) * g[i] * (inc + delta[i]);
        }
        s.psi += psiIncrement;
        s.previous = s.q;
        s.q = std::move(next);
    }
    State<R> original(const State<R> &s, const Vec<R> &g, const Vec<R> &force = {}) const {
        Vec<R> a(g.size()), b(g.size()), next(g.size());
        R aq = 0, ab = 0, an = 0, factor = R(dt * kappa);
        for (size_t i = 0; i < g.size(); ++i) {
            a[i] = R(.5) * factor * g[i];
            b[i] = a[i] / d[i];
            aq += a[i] * s.previous[i];
            ab += a[i] * b[i];
        }
        for (size_t i = 0; i < g.size(); ++i) {
            R bi = R((2 - dt * dt * omega[i]) / d[i]), ci = R((dt * sigma[i] - 1) / d[i]);
            next[i] = bi * s.q[i] + ci * s.previous[i] + b[i] * aq - factor * factor * s.psi * g[i] / d[i];
            if (!force.empty()) next[i] += dt * dt * force[i] / d[i];
            an += a[i] * next[i];
        }
        R psi = s.psi;
        for (size_t i = 0; i < g.size(); ++i) {
            next[i] -= b[i] * an / (1 + ab);
            psi += R(.5) * g[i] * (next[i] - s.previous[i]);
        }
        return {next, s.q, psi};
    }
};
