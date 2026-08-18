#pragma once

// Precomputes FluidSound's per-event-interval mass-matrix endpoint inverses on worker
// threads, ahead of the solver consuming them — with inverses, each RK4 stage's blended
// solve is two AMX matrix-vector products (see CoupledDirect::Solve).
//
// The schedule is static: trajectories are fixed at load, and the one dynamic input to
// Solver::Step's event bookkeeping (IsDead pruning) affects only the uncoupled list,
// which the mass matrix excludes. A cursor replays that bookkeeping exactly. Fetch
// verifies each interval's (t1, t2, n) and permanently falls back to inline computation
// on any mismatch, including the uncovered final event, whose interval endpoint the
// reference reads out of bounds.
//
// Adjacent intervals almost always share an endpoint time, their matrices differing
// only by the oscillators dying (rows/cols removed, order preserved) and born
// (appended) at that event. Ticket k's worker derives interval k+1's T1 inverse from
// its own fresh T2 inverse (see DeriveNextInverse): O(N²r) for churn r instead of
// O(N³), always one update from a fresh dpotrf/dpotri so error never compounds. This
// halves the load on the saturated matrix units, the throughput floor of bubble scenes.

#include "FluidSound.h"
#include "Parallel.h"
#include "Profile.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

struct BubbleFactorPipeline {
    BubbleFactorPipeline(FluidSound::Solver &solver, double dt, double ts)
        : Oscillators(&solver.Oscillators), Events(&solver.EventTimes),
          Integ(&solver.Integ), Dt(dt), Ts(ts) {
        if (Events->size() < 2) return;

        Integ->InverseProvider = [this](double t1, double t2, int n, Eigen::MatrixXd &i1, Eigen::MatrixXd &i2) { return Fetch(t1, t2, n, i1, i2); };
        // A handful of concurrent LAPACK calls saturate the matrix units' aggregate
        // throughput. Further workers (at any QoS — measured) only stretch every call,
        // including the main thread's own solve applies.
        unsigned n_workers = std::clamp(std::thread::hardware_concurrency() / 4u, 2u, 4u);
        if (const char *env = std::getenv("ACOUSTIC_BUBBLE_WORKERS")) n_workers = std::clamp(unsigned(atoi(env)), 1u, 16u);
        for (unsigned w = 0; w < n_workers; ++w) Workers.emplace_back([this] { WorkerLoop(); });
    }

    ~BubbleFactorPipeline() {
        Integ->InverseProvider = nullptr;
        {
            const std::scoped_lock lock{Mutex};
            Stop = true;
        }
        Cv.notify_all();
        for (auto &worker : Workers) worker.join();
        if (profile::Enabled() && StatIntervals > 0) {
            fprintf(stderr, "[bubble-pipeline] %llu intervals, mean N %.0f, worker busy %.1fs across %zu workers, %llu chained (mean churn %.1f), %llu chain fallbacks\n", static_cast<unsigned long long>(StatIntervals.load()), double(StatSumN) / double(StatIntervals), StatBusyNs * 1e-9, Workers.size(), static_cast<unsigned long long>(StatChained.load()), StatChained ? double(StatChurn) / double(StatChained) : 0., static_cast<unsigned long long>(StatChainFallbacks.load()));
            fprintf(stderr, "[bubble-pipeline] worker phases: mass %.1fs, factor %.1fs, symmetrize %.1fs, derive %.1fs (derive includes its own symmetrize)\n", StatMassNs * 1e-9, StatFactorNs * 1e-9, StatSymNs * 1e-9, StatDeriveNs * 1e-9);
        }
    }

private:
    using Osc = FluidSound::Oscillator;
    using Integrator = FluidSound::CoupledDirect;

    struct Interval {
        double T1, T2;
        std::vector<int> Coupled; // Oscillator indices, in the solver's coupled order
        bool ChainI1{false}; // T1 continues the previous interval's T2: I1 arrives from that ticket's worker
    };
    struct Inverses {
        Eigen::MatrixXd I1, I2; // Full symmetric endpoint inverses
        bool HasI1{false}, HasI2{false}; // Two producers: a chained I1 comes from ticket k-1's worker
        bool Ok{true}; // False: an endpoint inversion failed (not positive definite)
    };
    // Per-oscillator centers and radii at one endpoint time
    struct EndpointData {
        Eigen::ArrayXd Cx, Cy, Cz, R;
    };

    // Endpoint blend with the inline path's arithmetic. Bit-identical even when sd1
    // aliases sd2: at alpha 0 or 1 the inactive endpoint contributes exactly ±0.
    static void BlendEndpoint(const Eigen::Array<double, 6, Eigen::Dynamic> &sd1, const Eigen::Array<double, 6, Eigen::Dynamic> &sd2, double t1, double t2, double time, EndpointData &e) {
        const double alpha = (time - t1) / (t2 - t1);
        e.Cx = (1. - alpha) * sd1.row(2).transpose() + alpha * sd2.row(2).transpose();
        e.Cy = (1. - alpha) * sd1.row(3).transpose() + alpha * sd2.row(3).transpose();
        e.Cz = (1. - alpha) * sd1.row(4).transpose() + alpha * sd2.row(4).transpose();
        e.R = (1. - alpha) * sd1.row(0).transpose() + alpha * sd2.row(0).transpose();
    }

    // Mass matrix with CoupledDirect::ConstructMass's entry arithmetic, lower triangle only
    // (all Invert reads). Divide- and sqrt-throughput bound rather than matrix-unit bound, so
    // the column split runs alongside the other workers' LAPACK rather than competing with
    // it. Each entry is written once from read-only inputs, so the split is bit-identical.
    void ConstructMassLower(const EndpointData &e, Eigen::MatrixXd &m) const {
        const Timer timer{StatMassNs};
        const int n = int(e.R.size());
        m.resize(n, n);
        ParallelChunks(n, 64, [&](size_t begin, size_t end) {
            for (int i = int(begin); i < int(end); ++i) {
                m(i, i) = 1.;
                const int len = n - i - 1;
                if (len <= 0) continue;
                const auto dx = e.Cx.segment(i + 1, len) - e.Cx(i);
                const auto dy = e.Cy.segment(i + 1, len) - e.Cy(i);
                const auto dz = e.Cz.segment(i + 1, len) - e.Cz(i);
                m.col(i).segment(i + 1, len).array() = ((dx * dx + dy * dy + dz * dz) / (e.R(i) * e.R.segment(i + 1, len)) + Integrator::EpsSq).sqrt().inverse();
            }
        });
    }

    // Accumulates elapsed time into a shared counter, for the phase report below.
    struct Timer {
        explicit Timer(std::atomic<uint64_t> &acc) : Acc(acc), Start(std::chrono::steady_clock::now()) {}
        ~Timer() { Acc += uint64_t((std::chrono::steady_clock::now() - Start).count()); }
        std::atomic<uint64_t> &Acc;
        std::chrono::steady_clock::time_point Start;
    };

    // In-place SPD inversion (Cholesky) via Accelerate's AMX-backed LAPACK. Fills the lower
    // triangle; Symmetrize completes it. Returns false if the matrix is not positive definite.
    bool InvertLower(Eigen::MatrixXd &m) const {
        const Timer timer{StatFactorNs};
        const __LAPACK_int n = __LAPACK_int(m.rows());
        __LAPACK_int info = 0;
        if (n == 0) return true; // LAPACK rejects lda = 0
        dpotrf_("L", &n, m.data(), &n, &info);
        if (info != 0) return false;
        dpotri_("L", &n, m.data(), &n, &info); // Fills the lower triangle
        return info == 0;
    }

    // Mirrors the lower triangle into the upper: the dgemv apply and the chained
    // derivation's gathers both read full storage.
    void Symmetrize(Eigen::MatrixXd &m) const {
        const Timer timer{StatSymNs};
        m.triangularView<Eigen::StrictlyUpper>() = m.transpose();
    }

    bool Invert(Eigen::MatrixXd &m) const {
        const bool ok = InvertLower(m);
        Symmetrize(m);
        return ok;
    }

    // Event-dense stretches consume intervals faster than workers can produce them, so
    // quiet stretches must pre-fill deeply. The lookahead cap is by memory held in
    // unconsumed inverses, with a ticket floor so tiny systems never stall and a
    // ceiling to bound the bookkeeping.
    static constexpr size_t LookaheadBudgetBytes{size_t{3} << 29}; // 1.5 GB
    static constexpr int MinLookahead{4}, MaxLookahead{512};

    bool WantMoreTickets() const { // Called under Mutex
        if (NextTicket < Consumed + MinLookahead) return true;
        return InFlightBytes < LookaheadBudgetBytes && NextTicket < Consumed + MaxLookahead;
    }

    // Advances the cursor to the next refactor-triggering interval. Returns false at
    // schedule end. Called under Mutex.
    bool Advance() {
        const auto &events = *Events;
        const auto &oscs = *Oscillators;
        while (true) {
            const double time = Dt * CursorStep + Ts;
            if (time > events.back()) return false;
            while (CursorEv < events.size() && time >= events[CursorEv]) {
                if (CursorEv + 1 >= events.size()) return false; // Final event: leave uncovered
                if (time < events[CursorEv + 1]) {
                    const double t1 = events[CursorEv], t2 = events[CursorEv + 1];
                    std::erase_if(CursorCoupled, [&](int idx) { return t1 >= oscs[idx].EndTime; });
                    while (CursorOs < oscs.size() && time >= oscs[CursorOs].StartTime && oscs[CursorOs].StartTime < t2) {
                        if (t1 < oscs[CursorOs].EndTime) CursorCoupled.push_back(int(CursorOs));
                        ++CursorOs;
                    }
                    InFlightBytes += 2 * sizeof(double) * CursorCoupled.size() * CursorCoupled.size();
                    Pending.emplace(Generated, Interval{t1, t2, CursorCoupled, Generated > 0 && t1 == LastGeneratedT2});
                    LastGeneratedT2 = t2;
                    ++Generated;
                    ++CursorEv; // The re-checked crossing condition is false by construction
                    ++CursorStep;
                    return true;
                }
                ++CursorEv;
            }
            ++CursorStep;
        }
    }

    // Generates intervals until `upto` exist (or the schedule ends). Called under Mutex.
    void EnsureGenerated(int upto) {
        while (!GenerationDone && Generated < upto) {
            if (!Advance()) GenerationDone = true;
        }
    }

    // Derives inv(M(t2, next)) from h = inv(M(t2, cur)), where `next` removes cur's
    // dead oscillators (order preserved) and appends new ones: a delete-downdate via
    // the principal-submatrix identity inv(A_RR) = H_RR − H_RD inv(H_DD) H_DR, then a
    // bordered SPD extension via its Schur complement. Returns false when not derivable
    // (set mismatch, empty retained block, oversized churn, or a failed small Cholesky)
    // and the caller inverts from scratch.
    bool DeriveNextInverse(const Eigen::MatrixXd &h, const std::vector<int> &cur, const std::vector<int> &next, const EndpointData &e, double t2, std::vector<int> &cursors, Eigen::MatrixXd &out, int &churn) const {
        const Timer timer{StatDeriveNs};
        const int n = int(cur.size()), n2 = int(next.size());
        if (n == 0 || n2 == 0) return false;

        // Match next's retained prefix as an ordered subsequence of cur
        std::vector<int> ridx;
        ridx.reserve(n2);
        {
            int ci = 0;
            while (int(ridx.size()) < n2) {
                const int want = next[ridx.size()];
                while (ci < n && cur[ci] != want) ++ci;
                if (ci == n) break;
                ridx.push_back(ci++);
            }
        }
        const int m = int(ridx.size()), a = n2 - m, d = n - m;
        churn = a + d;
        if (m == 0 || 2 * (a + d) >= n2) return false;
        // The monotonic append cursor makes appended indices exceed every index in cur
        for (int j = m; j < n2; ++j) {
            if (next[j] <= cur[n - 1]) return false;
        }

        // The retained block is the result's top-left corner, so build it there directly,
        // with `out`'s leading dimension, rather than in scratch and copied across.
        out.resize(n2, n2);
        double *const p_data = out.data();
        const __LAPACK_int p_ld = n2;

        // Delete-downdate: the retained principal block's inverse, lower triangle valid
        if (d == 0) {
            out.topLeftCorner(m, m) = h;
        } else {
            std::vector<int> didx;
            didx.reserve(d);
            for (int ci = 0, ri = 0; ci < n; ++ci) {
                if (ri < m && ridx[ri] == ci) ++ri;
                else didx.push_back(ci);
            }
            Eigen::MatrixXd hdd(d, d), w(m, d);
            for (int j = 0; j < m; ++j) {
                const double *col = h.data() + size_t(ridx[j]) * n;
                double *dst = p_data + size_t(j) * p_ld;
                for (int i = 0; i < m; ++i) dst[i] = col[ridx[i]];
            }
            for (int j = 0; j < d; ++j) {
                const double *col = h.data() + size_t(didx[j]) * n;
                double *wc = w.data() + size_t(j) * m;
                for (int i = 0; i < m; ++i) wc[i] = col[ridx[i]];
                double *hc = hdd.data() + size_t(j) * d;
                for (int i = 0; i < d; ++i) hc[i] = col[didx[i]];
            }
            const __LAPACK_int dn = d;
            __LAPACK_int info = 0;
            dpotrf_("L", &dn, hdd.data(), &dn, &info);
            if (info != 0) return false;
            cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, m, d, 1., hdd.data(), d, w.data(), m);
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, m, d, -1., w.data(), m, 1., p_data, p_ld);
        }

        if (a == 0) {
            Symmetrize(out);
            return true;
        }

        // Bordered extension: retained-vs-new coupling u, new-vs-new block s
        Eigen::ArrayXd rx(m), ry(m), rz(m), rr(m);
        for (int i = 0; i < m; ++i) {
            rx(i) = e.Cx(ridx[i]);
            ry(i) = e.Cy(ridx[i]);
            rz(i) = e.Cz(ridx[i]);
            rr(i) = e.R(ridx[i]);
        }
        Eigen::ArrayXd ax(a), ay(a), az(a), ar(a);
        for (int j = 0; j < a; ++j) {
            const auto &osc = (*Oscillators)[next[m + j]];
            const auto sd = Osc::InterpAt(osc, t2, cursors[next[m + j]]);
            ar(j) = sd(0);
            ax(j) = sd(2);
            ay(j) = sd(3);
            az(j) = sd(4);
        }
        Eigen::MatrixXd u(m, a);
        for (int j = 0; j < a; ++j) {
            const auto dx = rx - ax(j);
            const auto dy = ry - ay(j);
            const auto dz = rz - az(j);
            u.col(j).array() = ((dx * dx + dy * dy + dz * dz) / (rr * ar(j)) + Integrator::EpsSq).sqrt().inverse();
        }
        Eigen::MatrixXd s(a, a);
        for (int j = 0; j < a; ++j) {
            s(j, j) = 1.;
            for (int i = j + 1; i < a; ++i) {
                const double dx = ax(i) - ax(j), dy = ay(i) - ay(j), dz = az(i) - az(j);
                s(i, j) = 1. / std::sqrt((dx * dx + dy * dy + dz * dz) / (ar(i) * ar(j)) + Integrator::EpsSq);
                s(j, i) = s(i, j);
            }
        }

        Eigen::MatrixXd x2(m, a);
        cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, m, a, 1., p_data, p_ld, u.data(), m, 0., x2.data(), m);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, a, a, m, -1., u.data(), m, x2.data(), m, 1., s.data(), a);
        const __LAPACK_int an = a;
        __LAPACK_int info = 0;
        dpotrf_("L", &an, s.data(), &an, &info); // s holds the Schur complement's Cholesky factor
        if (info != 0) return false;
        Eigen::MatrixXd v = x2;
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, m, a, 1., s.data(), a, v.data(), m);
        cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, m, a, 1., v.data(), m, 1., p_data, p_ld); // Top-left, lower
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, m, a, -1., s.data(), a, v.data(), m); // v = top-right
        dpotri_("L", &an, s.data(), &an, &info); // s = inv(Schur complement), lower
        if (info != 0) return false;

        out.block(m, 0, a, m) = v.transpose();
        out.bottomRightCorner(a, a).triangularView<Eigen::Lower>() = s.triangularView<Eigen::Lower>();
        Symmetrize(out);
        return true;
    }

    void WorkerLoop() {
        std::vector<int> cursors(Oscillators->size(), 0); // Per-oscillator interp cursors
        while (true) {
            const Interval *interval{};
            const Interval *next{};
            int k;
            {
                std::unique_lock lock{Mutex};
                Cv.wait(lock, [&] { return Stop || Exhausted || WantMoreTickets(); });
                if (Stop || Exhausted) return;
                EnsureGenerated(NextTicket + 1);
                if (Generated <= NextTicket) { // Tickets issue in order: the schedule is spent
                    Exhausted = true;
                    lock.unlock();
                    Cv.notify_all();
                    return;
                }
                k = NextTicket++;
                EnsureGenerated(k + 2); // Peek at the successor for the chained derivation
                interval = &Pending.at(k); // Map nodes are stable across later inserts/erases
                if (auto it = Pending.find(k + 1); it != Pending.end()) next = &it->second;
            }
            const auto busy_start = std::chrono::steady_clock::now();
            const int n = int(interval->Coupled.size());
            Eigen::Array<double, 6, Eigen::Dynamic> sd1, sd2(6, n);
            for (int i = 0; i < n; ++i) sd2.col(i) = Osc::InterpAt((*Oscillators)[interval->Coupled[i]], interval->T2, cursors[interval->Coupled[i]]);

            Inverses inverses;
            bool ok1 = true;
            if (!interval->ChainI1) {
                sd1.resize(6, n);
                for (int i = 0; i < n; ++i) sd1.col(i) = Osc::InterpAt((*Oscillators)[interval->Coupled[i]], interval->T1, cursors[interval->Coupled[i]]);
                EndpointData e1;
                BlendEndpoint(sd1, sd2, interval->T1, interval->T2, interval->T1, e1);
                ConstructMassLower(e1, inverses.I1);
                ok1 = Invert(inverses.I1);
            }
            EndpointData e2;
            BlendEndpoint(sd1.size() ? sd1 : sd2, sd2, interval->T1, interval->T2, interval->T2, e2);
            ConstructMassLower(e2, inverses.I2);
            const bool ok2 = Invert(inverses.I2);

            // The successor's T1 inverse, derived from this fresh T2 inverse (same time)
            Eigen::MatrixXd derived;
            bool derived_ok = false;
            if (next && next->ChainI1) {
                int churn = 0;
                derived_ok = ok2 && DeriveNextInverse(inverses.I2, interval->Coupled, next->Coupled, e2, interval->T2, cursors, derived, churn);
                if (derived_ok) {
                    StatChained += 1;
                    StatChurn += uint64_t(churn);
                } else {
                    StatChainFallbacks += 1;
                    const int n2 = int(next->Coupled.size());
                    EndpointData en;
                    en.Cx.resize(n2);
                    en.Cy.resize(n2);
                    en.Cz.resize(n2);
                    en.R.resize(n2);
                    for (int i = 0; i < n2; ++i) {
                        const auto sd = Osc::InterpAt((*Oscillators)[next->Coupled[i]], interval->T2, cursors[next->Coupled[i]]);
                        en.R(i) = sd(0);
                        en.Cx(i) = sd(2);
                        en.Cy(i) = sd(3);
                        en.Cz(i) = sd(4);
                    }
                    ConstructMassLower(en, derived);
                    derived_ok = Invert(derived);
                }
            }
            StatBusyNs += (std::chrono::steady_clock::now() - busy_start).count();
            StatIntervals += 1;
            StatSumN += n;
            {
                const std::scoped_lock lock{Mutex};
                auto &ready = Ready[k];
                ready.I2 = std::move(inverses.I2);
                ready.HasI2 = true;
                ready.Ok &= ok2;
                if (!interval->ChainI1) {
                    ready.I1 = std::move(inverses.I1);
                    ready.HasI1 = true;
                    ready.Ok &= ok1;
                }
                if (next && next->ChainI1) {
                    auto &ready_next = Ready[k + 1];
                    ready_next.I1 = std::move(derived);
                    ready_next.HasI1 = true;
                    ready_next.Ok &= derived_ok;
                }
            }
            Cv.notify_all();
        }
    }

    bool Fetch(double t1, double t2, int n, Eigen::MatrixXd &i1, Eigen::MatrixXd &i2) {
        const profile::Scope scope{"bubble/fetch"}; // Mostly waiting on workers
        std::unique_lock lock{Mutex};
        const int k = Consumed;
        const auto complete = [&] {
            const auto it = Ready.find(k);
            return it != Ready.end() && it->second.HasI1 && it->second.HasI2;
        };
        Cv.wait(lock, [&] { return Stop || complete() || (Exhausted && k >= Generated); });
        const auto it = Pending.find(k);
        if (Stop || !complete() || !Ready.at(k).Ok || it->second.T1 != t1 || it->second.T2 != t2 || int(it->second.Coupled.size()) != n) {
            Stop = true; // Schedule diverged, non-PD endpoint, or uncovered tail: inline from here on
            lock.unlock();
            Cv.notify_all();
            return false;
        }
        auto node = Ready.extract(k);
        i1 = std::move(node.mapped().I1);
        i2 = std::move(node.mapped().I2);
        InFlightBytes -= 2 * sizeof(double) * it->second.Coupled.size() * it->second.Coupled.size();
        Pending.erase(it);
        Consumed = k + 1;
        lock.unlock();
        Cv.notify_all(); // Wake workers gated on the lookahead window
        return true;
    }

    std::vector<Osc> *Oscillators;
    const std::vector<double> *Events;
    Integrator *Integ{nullptr};
    double Dt, Ts;

    // Schedule cursor state (see Advance)
    long CursorStep{0};
    size_t CursorEv{0}, CursorOs{0};
    std::vector<int> CursorCoupled;
    double LastGeneratedT2{0.};

    std::mutex Mutex;
    std::condition_variable Cv;
    std::map<int, Interval> Pending; // Generated, not yet consumed
    std::map<int, Inverses> Ready;
    size_t InFlightBytes{0}; // Memory held by unconsumed inverses
    int NextTicket{0}, Generated{0}, Consumed{0};
    bool Stop{false}, Exhausted{false}, GenerationDone{false};
    std::vector<std::thread> Workers;

    // Worker-side stats, reported at teardown under ACOUSTIC_PROFILE
    std::atomic<uint64_t> StatBusyNs{0}, StatIntervals{0}, StatSumN{0};
    std::atomic<uint64_t> StatChained{0}, StatChurn{0}, StatChainFallbacks{0};
    // Per-phase worker time. Derive does its own mirror, so it overlaps StatSymNs.
    mutable std::atomic<uint64_t> StatMassNs{0}, StatFactorNs{0}, StatSymNs{0}, StatDeriveNs{0};
};
