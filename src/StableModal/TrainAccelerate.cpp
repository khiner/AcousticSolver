#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <dispatch/dispatch.h>
#include <exception>
#include <libproc.h>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace {
double sign(double x) { return (x > 0) - (x < 0); }
struct SingleThreadedBLAS {
    decltype(BLASGetThreading()) previous = BLASGetThreading();
    SingleThreadedBLAS() { BLASSetThreading(BLAS_THREADING_SINGLE_THREADED); }
    ~SingleThreadedBLAS() { BLASSetThreading(previous); }
};
} // namespace

struct Forward {
    static constexpr unsigned m = 75, dim = 151;
    unsigned h, count, stride;
    double k, halfStep, lambda;
    const double *weights, *transposed, *bias, *scale, *physical;
    std::vector<double> velocity, zq, zp, lv;
    bool advanced = false;

    Forward(unsigned h, unsigned count, double fs, double lambda, const double *weights, const double *transposed, const double *bias, const double *scale, const double *physical)
        : h(h), count(count), stride(150 + 4 * h), k(1 / fs), halfStep(.5 * k), lambda(lambda), weights(weights),
          transposed(transposed), bias(bias), scale(scale), physical(physical), velocity(count * m), zq(count * h), zp(count * h), lv(count) {}
    void hidden(const double *x, unsigned sx, double *z, unsigned sz) const {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, count, h, m, 1., x, sx, transposed, h, 0., z, sz);
    }
    void modes(const double *z, unsigned sz, double *x, unsigned sx, double beta = 0.) const {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, count, m, h, 1., z, sz, weights, m, beta, x, sx);
    }

    void initialize(const double *initial, unsigned initialWidth, double *output, unsigned outputStride, double *tape) {
        for (unsigned b = 0; b < count; ++b) {
            std::copy_n(initial + b * initialWidth, 2 * m, output + b * outputStride);
            std::copy_n(initial + b * initialWidth, m, tape + b * stride);
        }
        hidden(tape, stride, tape + 2 * m, stride);
        for (unsigned b = 0; b < count; ++b) {
            double *row = tape + b * stride, v = 0;
            for (unsigned j = 0; j < h; ++j) {
                zq[b * h + j] = row[2 * m + j];
                double z = zq[b * h + j] + bias[j], a = scale[j] * (z > 0 ? 1. : .01) * z;
                row[2 * m + j] = z;
                row[2 * m + h + j] = a;
                v += .5 * a * z;
            }
            output[b * outputStride + 2 * m] = initialWidth == dim ? initial[b * initialWidth + 2 * m] : std::sqrt(2 * v + 1e-12);
        }
    }

    void step(const double *input, unsigned inputStride, double *output, unsigned outputStride, const double *excitation, unsigned excitationStride, double *t, double *f, double *psiH) {
        for (unsigned b = 0; b < count; ++b) {
            const double *y = input + b * inputStride;
            for (unsigned i = 0; i < m; ++i) {
                t[b * stride + i] = y[i] + halfStep * y[m + i];
                velocity[b * m + i] = y[m + i];
            }
        }
        // Advance the linear projection alongside q, with one W*p product per step.
        hidden(velocity.data(), m, zp.data(), h);
        for (unsigned b = 0; b < count; ++b) {
            double *row = t + b * stride, vh = 0, vq = 0;
            for (unsigned j = 0; j < h; ++j) {
                double const z0 = zq[b * h + j] + (advanced ? halfStep * zp[b * h + j] : 0.) + bias[j];
                zq[b * h + j] += (advanced ? k : halfStep) * zp[b * h + j];
                double const z = zq[b * h + j] + bias[j];
                double const a = scale[j] * (z > 0 ? 1. : .01) * z;
                row[2 * m + j] = z;
                row[2 * m + h + j] = a;
                vh += .5 * a * z;
                vq += .5 * scale[j] * (z0 > 0 ? 1. : .01) * z0 * z0;
            }
            psiH[b] = std::sqrt(2 * vh + 1e-12);
            lv[b] = std::sqrt(2 * vq + 1e-12);
        }
        modes(t + 2 * m + h, stride, f, m);
        for (unsigned b = 0; b < count; ++b) {
            const double *y = input + b * inputStride, *ph = physical + b * 226;
            double *yn = output + b * outputStride, *row = t + b * stride;
            double norm = 0, ap = 0, br = 0, ab = 0, mu = ph[225], a[75], bv[75], rhs[75];
            for (unsigned i = 0; i < m; ++i) norm += std::abs(y[m + i]);
            for (unsigned i = 0; i < m; ++i) {
                double const g = f[b * m + i] / psiH[b] - lambda * (y[150] - lv[b]) * sign(y[m + i]) / (norm + 1e-12);
                row[m + i] = g;
                a[i] = halfStep * mu * g;
                bv[i] = a[i] / (1 + k * ph[m + i]);
                ap += a[i] * y[m + i];
            }
            for (unsigned i = 0; i < m; ++i) {
                rhs[i] = (1 - k * ph[m + i]) * y[m + i] - a[i] * ap + k * (-ph[i] * row[i] - mu * mu * row[m + i] * y[150] + ph[2 * m + i] * excitation[b * excitationStride]);
                br += bv[i] * rhs[i];
                ab += a[i] * bv[i];
            }
            double increment = 0;
            for (unsigned i = 0; i < m; ++i) {
                yn[m + i] = rhs[i] / (1 + k * ph[m + i]) - bv[i] * br / (1 + ab);
                yn[i] = row[i] + halfStep * yn[m + i];
                increment += row[m + i] * .5 * (y[m + i] + yn[m + i]);
            }
            yn[150] = y[150] + k * increment;
        }
        advanced = true;
    }
};

static std::vector<double> transposeWeights(unsigned h, const double *weights) {
    std::vector<double> transposed(75 * h);
    for (unsigned j = 0; j < h; ++j)
        for (unsigned i = 0; i < 75; ++i) transposed[i * h + j] = weights[j * 75 + i];
    return transposed;
}

struct Stream {
    std::vector<double> weights, bias, scale, physical, transposed, state, next, tape, force, psiH;
    Forward forward;

    Stream(unsigned h, unsigned count, unsigned initialWidth, double fs, double lambda, const double *w, const double *b, const double *s, const double *ph, const double *initial)
        : weights(w, w + 75 * h), bias(b, b + h), scale(s, s + h), physical(ph, ph + 226 * count),
          transposed(transposeWeights(h, w)), state(count * 151), next(count * 151), tape(count * (150 + 4 * h)),
          force(count * 75), psiH(count),
          forward(h, count, fs, lambda, weights.data(), transposed.data(), bias.data(), scale.data(), physical.data()) {
        forward.initialize(initial, initialWidth, state.data(), 151, tape.data());
    }

    void process(unsigned frames, const double *excitation, double *output) {
        for (unsigned n = 0; n < frames; ++n) {
            for (unsigned b = 0; b < forward.count; ++b)
                std::copy_n(state.data() + b * 151, 151, output + (size_t(b) * frames + n) * 151);
            forward.step(state.data(), 151, next.data(), 151, excitation + n, frames, tape.data(), force.data(), psiH.data());
            state.swap(next);
        }
        for (double const x : state)
            if (!std::isfinite(x)) throw std::runtime_error("Nonfinite Accelerate render state");
    }
};

extern "C" int modal_stream_create(unsigned h, unsigned count, unsigned initialWidth, double fs, double lambda, const double *weights, const double *bias, const double *scale, const double *physical, const double *initial, void **handle, char *error, size_t capacity) {
    try {
        if (!h || h > 1000 || !count || count > 2048 || (initialWidth != 150 && initialWidth != 151) || !std::isfinite(fs) || fs <= 0 || !std::isfinite(lambda))
            throw std::runtime_error("Invalid Accelerate render dimensions or sample rate");
        *handle = new Stream(h, count, initialWidth, fs, lambda, weights, bias, scale, physical, initial);
        return 0;
    } catch (const std::exception &e) {
        std::snprintf(error, capacity, "%s", e.what());
        return 1;
    }
}

extern "C" int modal_stream_process(void *handle, unsigned frames, const double *excitation, double *output, char *error, size_t capacity) {
    if (!handle) {
        std::snprintf(error, capacity, "Closed Accelerate renderer");
        return 1;
    }
    SingleThreadedBLAS const threading;
    try {
        static_cast<Stream *>(handle)->process(frames, excitation, output);
        return 0;
    } catch (const std::exception &e) {
        std::snprintf(error, capacity, "%s", e.what());
        return 1;
    }
}

extern "C" void modal_stream_destroy(void *handle) { delete static_cast<Stream *>(handle); }

template<bool backward>
static void evaluate(unsigned h, unsigned count, unsigned frames, double fs, double lambda, const double *weights, const double *transposed, const double *bias, const double *scale, const double *scaleIn, const double *physical, const double *excitation, const double *initial, const double *target, double *output, double *gradient, double *initialGradient, double *loss) {
    const unsigned m = 75, dim = 151, stride = 2 * m + 4 * h, rows = count * frames;
    const double k = 1 / fs, halfStep = .5 * k, inv = 1. / (count * frames * m);
    const unsigned savedRows = backward ? rows : count;
    auto const tape = std::make_unique_for_overwrite<double[]>(size_t(savedRows) * stride);
    auto const force = std::make_unique_for_overwrite<double[]>(size_t(savedRows) * m);
    auto const psiH = std::make_unique_for_overwrite<double[]>(savedRows);
    std::fill_n(tape.get(), size_t(count) * stride, 0.);
    Forward forward(h, count, fs, lambda, weights, transposed, bias, scale, physical);
    auto &lv = forward.lv, &zq = forward.zq;
    forward.initialize(initial, 150, output, frames * dim, tape.get());
    for (unsigned n = 0; n < frames - 1; ++n) {
        const unsigned savedStep = backward ? n + 1 : 0;
        double *t = tape.get() + size_t(savedStep) * count * stride;
        double *f = force.get() + size_t(savedStep) * count * m;
        forward.step(output + n * dim, frames * dim, output + (n + 1) * dim, frames * dim, excitation + n, frames, t, f, psiH.get() + savedStep * count);
    }
    *loss = 0;
    // Reduce the loss once, in the same order with or without gradients.
    for (unsigned n = frames; n-- > 0;)
        for (unsigned b = 0; b < count; ++b)
            for (unsigned i = 0; i < m; ++i) {
                const size_t offset = (b * frames + n) * dim + i;
                double eq = output[offset] - target[offset], ep = output[offset + m] - target[offset + m];
                *loss += (eq * eq + ep * ep) * inv;
            }
    if (!std::isfinite(*loss)) throw std::runtime_error("Nonfinite Accelerate loss");
    if constexpr (!backward) return;
    std::vector<double> lq(count * m), lp(count * m), lpsi(count);
    for (unsigned back = 0; back < frames - 1; ++back) {
        unsigned const n = frames - 2 - back;
        double *t = tape.get() + size_t(n + 1) * count * stride;
        for (unsigned b = 0; b < count; ++b) {
            const double *y = output + (b * frames + n) * dim, *yn = y + dim, *truth = target + (b * frames + n + 1) * dim, *ph = physical + b * 226;
            double *row = t + b * stride, *bq = lq.data() + b * m, *bp = lp.data() + b * m;
            double a[75], bv[75], rhs[75], ln[75], lr[75], la[75], ap = 0, br = 0, ab = 0, lc = 0, lap = 0, lgf = 0, lpg = 0, mu = ph[225];
            for (unsigned i = 0; i < m; ++i) {
                double eq = yn[i] - truth[i], ep = yn[m + i] - truth[m + i];
                bq[i] += 2 * inv * eq;
                bp[i] += 2 * inv * ep;
                a[i] = halfStep * mu * row[m + i];
                bv[i] = a[i] / (1 + k * ph[m + i]);
                ap += a[i] * y[m + i];
            }
            for (unsigned i = 0; i < m; ++i) {
                rhs[i] = (1 - k * ph[m + i]) * y[m + i] - a[i] * ap + k * (-ph[i] * row[i] - mu * mu * row[m + i] * y[150] + ph[2 * m + i] * excitation[b * frames + n]);
                br += bv[i] * rhs[i];
                ab += a[i] * bv[i];
                ln[i] = bp[i] + halfStep * bq[i] + halfStep * lpsi[b] * row[m + i];
                lc -= ln[i] * bv[i];
            }
            double lbr = lc / (1 + ab), lab = -lc * br / ((1 + ab) * (1 + ab));
            for (unsigned i = 0; i < m; ++i) {
                double dp = 1 + k * ph[m + i], lb = -ln[i] * br / (1 + ab) + lbr * rhs[i] + lab * a[i];
                lr[i] = ln[i] / dp + lbr * bv[i];
                la[i] = lab * bv[i] + lb / dp - lr[i] * ap;
                lap -= lr[i] * a[i];
                lpg += lr[i] * row[m + i];
            }
            for (unsigned i = 0; i < m; ++i) {
                double g = row[m + i], lg = halfStep * lpsi[b] * (y[m + i] + yn[m + i]) - k * mu * mu * y[150] * lr[i] + halfStep * mu * (la[i] + lap * y[m + i]);
                bp[i] = halfStep * lpsi[b] * g + (1 - k * ph[m + i]) * lr[i] + lap * a[i];
                bq[i] -= k * ph[i] * lr[i];
                row[m + i] = lg / psiH[(n + 1) * count + b];
                lgf += lg * force[((n + 1) * count + b) * m + i];
            }
            double const psi = psiH[(n + 1) * count + b];
            lv[b] = -lgf / (psi * psi * psi);
            lpsi[b] -= k * mu * mu * lpg;
        }
        forward.hidden(t + m, stride, zq.data(), h);
        for (unsigned b = 0; b < count; ++b)
            for (unsigned j = 0; j < h; ++j) {
                double *row = t + b * stride, z = row[2 * m + j], a = row[2 * m + h + j], dot = zq[b * h + j];
                double const dz = scale[j] * (z > 0 ? 1. : .01) * dot + lv[b] * a;
                row[2 * m + j] = dz;
                row[2 * m + 2 * h + j] = a * dot + .5 * lv[b] * a * z;
                row[2 * m + 3 * h + j] = dz * (z - bias[j]) + a * dot;
            }
        forward.modes(t + 2 * m, stride, lq.data(), m, 1.);
        for (unsigned i = 0; i < count * m; ++i) lp[i] += halfStep * lq[i];
    }
    for (unsigned b = 0; b < count; ++b) {
        double *row = tape.get() + b * stride;
        double const v = lpsi[b] / output[b * frames * dim + 150];
        for (unsigned j = 0; j < h; ++j) {
            double z = row[2 * m + j], dz = v * row[2 * m + h + j];
            row[2 * m + j] = dz;
            row[2 * m + h + j] = 0;
            row[2 * m + 2 * h + j] = .5 * dz * z;
            row[2 * m + 3 * h + j] = dz * (z - bias[j]);
        }
    }
    forward.modes(tape.get() + 2 * m, stride, lq.data(), m, 1.);
    for (unsigned b = 0; b < count; ++b)
        for (unsigned i = 0; i < m; ++i) {
            double const eq = output[b * frames * dim + i] - target[b * frames * dim + i];
            double const ep = output[b * frames * dim + m + i] - target[b * frames * dim + m + i];
            initialGradient[b * 2 * m + i] = lq[b * m + i] + 2 * inv * eq;
            initialGradient[b * 2 * m + m + i] = lp[b * m + i] + 2 * inv * ep;
        }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, h, m, rows, 1., tape.get() + 2 * m, stride, tape.get(), stride, 0., gradient, m);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, h, m, rows, 1., tape.get() + 2 * m + h, stride, tape.get() + m, stride, 1., gradient, m);
    for (unsigned j = 0; j < h; ++j)
        for (unsigned i = 0; i < m; ++i) gradient[j * m + i] *= scaleIn[j];
    std::fill_n(gradient + h * m, 3 * h, 0.);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned j = 0; j < h; ++j) {
            const double *row = tape.get() + size_t(r) * stride;
            gradient[h * m + j] += row[2 * m + j];
            gradient[h * m + h + j] += row[2 * m + 3 * h + j];
            gradient[h * m + 2 * h + j] += row[2 * m + 2 * h + j];
        }
    for (unsigned i = 0; i < 78 * h; ++i)
        if (!std::isfinite(gradient[i])) throw std::runtime_error("Nonfinite Accelerate gradient");
}

extern "C" int modal_evaluate(unsigned h, unsigned count, unsigned frames, double fs, double lambda, const double *weights, const double *bias, const double *scale, const double *scaleIn, const double *physical, const double *excitation, const double *initial, const double *target, double *output, double *gradient, double *initialGradient, double *loss, char *error, size_t capacity) {
    try {
        if (!h || h > 1000 || !count || count > 2048 || frames < 2 || frames > 128 || fs <= 0 || (!gradient != !initialGradient))
            throw std::runtime_error("Invalid Accelerate training dimensions");
        auto transposed = transposeWeights(h, weights);
        const double *wt = transposed.data();
        const bool backward = gradient != nullptr;
        auto invoke = backward ? evaluate<true> : evaluate<false>;
        const unsigned tile = count <= 16 ? 1 : (count < 128 ? 8 : 16);
        if (count <= tile) {
            invoke(h, count, frames, fs, lambda, weights, wt, bias, scale, scaleIn, physical, excitation, initial, target, output, gradient, initialGradient, loss);
            return 0;
        }
        const unsigned groups = (count + tile - 1) / tile, parameters = 78 * h;
        std::vector<double> gradients(backward ? size_t(groups) * parameters : 0), losses(groups);
        std::vector<std::exception_ptr> errors(groups);
        auto *g = gradients.data(), *l = losses.data();
        auto *e = errors.data();
        dispatch_apply(groups, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t group) {
          unsigned offset = unsigned(group) * tile, n = std::min(tile, count - offset);
          SingleThreadedBLAS const threading;
          try {
              invoke(h, n, frames, fs, lambda, weights, wt, bias, scale, scaleIn, physical + offset * 226, excitation + offset * frames, initial + offset * 150, target + offset * frames * 151, output + offset * frames * 151, backward ? g + group * parameters : nullptr, backward ? initialGradient + offset * 150 : nullptr, l + group);
          } catch (...) { e[group] = std::current_exception(); }
        });
        if (backward) std::fill_n(gradient, parameters, 0.);
        *loss = 0;
        for (unsigned group = 0; group < groups; ++group) {
            if (errors[group]) std::rethrow_exception(errors[group]);
            unsigned offset = group * tile, n = std::min(tile, count - offset);
            double const fraction = double(n) / count;
            *loss += fraction * losses[group];
            if (backward) {
                for (unsigned i = 0; i < parameters; ++i) gradient[i] += fraction * gradients[group * parameters + i];
                for (unsigned i = 0; i < n * 150; ++i) initialGradient[offset * 150 + i] *= fraction;
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::snprintf(error, capacity, "%s", e.what());
        return 1;
    }
}

extern "C" int modal_adam(unsigned count, unsigned step, double lr, double beta1, double beta2, double epsilon, double *parameters, const double *gradient, double *first, double *second, char *error, size_t capacity) {
    if (!count || !step || lr <= 0 || beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || epsilon <= 0) {
        std::snprintf(error, capacity, "Invalid Adam settings");
        return 1;
    }
    const double stepSize = lr / (1 - std::pow(beta1, step)), correction2 = std::sqrt(1 - std::pow(beta2, step));
    for (unsigned i = 0; i < count; ++i) {
        first[i] = beta1 * first[i] + (1 - beta1) * gradient[i];
        second[i] = beta2 * second[i] + (1 - beta2) * gradient[i] * gradient[i];
        parameters[i] -= stepSize * first[i] / (std::sqrt(second[i]) / correction2 + epsilon);
        if (!std::isfinite(parameters[i]) || !std::isfinite(first[i]) || !std::isfinite(second[i])) {
            std::snprintf(error, capacity, "Nonfinite Accelerate Adam output");
            return 1;
        }
    }
    return 0;
}

extern "C" int modal_training_memory(uint64_t *values) {
    rusage_info_v4 usage{};
    int const result = proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t *>(&usage));
    values[0] = usage.ri_phys_footprint;
    values[1] = usage.ri_lifetime_max_phys_footprint;
    return result;
}
