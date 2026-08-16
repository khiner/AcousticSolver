#pragma once

// Precomputes FluidSound's per-event-interval mass-matrix endpoint INVERSES on
// background worker threads, ahead of the solver consuming them. With the inverses in
// hand, every RK4 stage's blended solve is two small matrix-vector products instead of
// two pairs of triangular substitutions (see Coupled_Direct::solve) — the dominant
// main-thread cost of bubble scenes.
//
// The schedule is static: oscillator trajectories are fixed at load, and event times
// quantize onto the solver's fixed step grid. The one dynamic input to Solver::step's
// event bookkeeping (is_dead pruning) touches only the uncoupled list, which the mass
// matrix excludes, so a cursor replays that bookkeeping exactly, generating intervals
// lazily as workers take tickets. Fetch verifies each interval's (t1, t2, N) against the
// schedule and permanently falls back to inline computation on any mismatch, including
// the uncovered final event, whose interval endpoint the reference reads out of bounds.

#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <vector>

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include "FluidSound.h"
#include "Profile.h"

class BubbleFactorPipeline {
    using Osc = FluidSound::Oscillator<double>;
    using Integrator = FluidSound::Coupled_Direct<double>;

    struct Interval {
        double T1, T2;
        std::vector<int> Coupled; // oscillator indices, in the solver's coupled order
    };
    struct Inverses {
        Eigen::MatrixXd I1, I2; // full symmetric endpoint inverses
        bool Ok{false}; // both endpoint matrices were positive definite
    };

    // Endpoint mass matrix (see Coupled_Direct::ConstructMass), built serially with
    // vectorized column fills, and only the lower triangle (all Invert reads). Serial
    // because each worker builds its own matrix — the inline path's parallel fill would
    // thrash the dispatch pool when invoked from every worker at once.
    static void ConstructMassLower(const Eigen::Array<double, 6, Eigen::Dynamic> &sd1, const Eigen::Array<double, 6, Eigen::Dynamic> &sd2, double t1, double t2, double time, Eigen::MatrixXd &m) {
        const double alpha = (time - t1) / (t2 - t1);
        const int n = int(sd1.cols());
        const Eigen::ArrayXd cx = (1. - alpha) * sd1.row(2).transpose() + alpha * sd2.row(2).transpose();
        const Eigen::ArrayXd cy = (1. - alpha) * sd1.row(3).transpose() + alpha * sd2.row(3).transpose();
        const Eigen::ArrayXd cz = (1. - alpha) * sd1.row(4).transpose() + alpha * sd2.row(4).transpose();
        const Eigen::ArrayXd r = (1. - alpha) * sd1.row(0).transpose() + alpha * sd2.row(0).transpose();

        m.resize(n, n);
        for (int i = 0; i < n; ++i) {
            m(i, i) = 1.;
            const int len = n - i - 1;
            if (len <= 0) continue;
            const auto dx = cx.segment(i + 1, len) - cx(i);
            const auto dy = cy.segment(i + 1, len) - cy(i);
            const auto dz = cz.segment(i + 1, len) - cz(i);
            m.col(i).segment(i + 1, len).array() = ((dx * dx + dy * dy + dz * dz) / (r(i) * r.segment(i + 1, len)) + Integrator::EpsSq).sqrt().inverse();
        }
    }

    // In-place SPD inversion (Cholesky) via Accelerate's AMX-backed LAPACK, then
    // symmetrized to full storage (the solver applies it with cblas_dgemv). Returns
    // false if the matrix is not positive definite.
    static bool Invert(Eigen::MatrixXd &m) {
        __LAPACK_int n = __LAPACK_int(m.rows()), info = 0;
        if (n == 0) return true; // LAPACK rejects lda = 0
        dpotrf_("L", &n, m.data(), &n, &info);
        if (info != 0) return false;
        dpotri_("L", &n, m.data(), &n, &info); // fills the lower triangle
        m.triangularView<Eigen::StrictlyUpper>() = m.transpose();
        return info == 0;
    }

public:
    BubbleFactorPipeline(FluidSound::Solver<double> &solver, double dt, double ts)
        : Oscillators(&solver.oscillators()), Events(&solver.eventTimes()),
          Integ(dynamic_cast<Integrator *>(solver.integrator())), Dt(dt), Ts(ts) {
        if (!Integ || Events->size() < 2) return;

        Integ->InverseProvider = [this](double t1, double t2, int n, Eigen::MatrixXd &i1, Eigen::MatrixXd &i2) { return Fetch(t1, t2, n, i1, i2); };
        // The inversions are matrix-unit-bound: a handful of concurrent LAPACK calls
        // saturate the units' aggregate throughput, and further workers (at any QoS —
        // measured) only stretch every call, including the main thread's own solve
        // applies. The deep lookahead (see WantMoreTickets) keeps the small pool busy
        // through event-dense stretches.
        const unsigned n_workers = std::clamp(std::thread::hardware_concurrency() / 4u, 2u, 4u);
        for (unsigned w = 0; w < n_workers; ++w) Workers.emplace_back([this] { WorkerLoop(); });
    }

    ~BubbleFactorPipeline() {
        if (Integ) Integ->InverseProvider = nullptr;
        {
            const std::scoped_lock lock{Mutex};
            Stop = true;
        }
        Cv.notify_all();
        for (auto &worker : Workers) worker.join();
        if (profile::Enabled() && StatIntervals > 0) {
            fprintf(stderr, "[bubble-pipeline] %llu intervals, mean N %.0f, worker busy %.1fs across %zu workers\n", static_cast<unsigned long long>(StatIntervals.load()), double(StatSumN) / double(StatIntervals), StatBusyNs * 1e-9, Workers.size());
        }
    }

private:
    // Workers may run far ahead of consumption — event-dense stretches consume intervals
    // much faster than they can be produced, so quiet stretches must pre-fill the buffer.
    // The cap is by memory held in not-yet-consumed inverses, with a small ticket floor
    // so tiny systems never stall, and a ticket ceiling to bound the bookkeeping.
    static constexpr size_t LookaheadBudgetBytes = size_t{3} << 29; // 1.5 GB
    static constexpr int MinLookahead{4}, MaxLookahead{512};

    bool WantMoreTickets() const { // called under Mutex
        if (NextTicket < Consumed + MinLookahead) return true;
        return InFlightBytes < LookaheadBudgetBytes && NextTicket < Consumed + MaxLookahead;
    }

    // Advances the schedule cursor to the next refactor-triggering interval, appending it
    // to Pending. Returns false at schedule end. Called under Mutex.
    bool Advance() {
        const auto &events = *Events;
        const auto &oscs = *Oscillators;
        while (true) {
            const double time = Dt * CursorStep + Ts;
            if (time > events.back()) return false;
            while (CursorEv < events.size() && time >= events[CursorEv]) {
                if (CursorEv + 1 >= events.size()) return false; // final event: leave uncovered
                if (time < events[CursorEv + 1]) {
                    const double t1 = events[CursorEv], t2 = events[CursorEv + 1];
                    std::erase_if(CursorCoupled, [&](int idx) { return t1 >= oscs[idx].endTime; });
                    while (CursorOs < oscs.size() && time >= oscs[CursorOs].startTime && oscs[CursorOs].startTime < t2) {
                        if (t1 < oscs[CursorOs].endTime) CursorCoupled.push_back(int(CursorOs));
                        ++CursorOs;
                    }
                    InFlightBytes += 2 * sizeof(double) * CursorCoupled.size() * CursorCoupled.size();
                    Pending.emplace(Generated++, Interval{t1, t2, CursorCoupled});
                    ++CursorEv; // the re-checked crossing condition is false by construction
                    ++CursorStep;
                    return true;
                }
                ++CursorEv;
            }
            ++CursorStep;
        }
    }

    void WorkerLoop() {
        std::vector<int> cursors(Oscillators->size(), 0); // per-oscillator interp cursors
        while (true) {
            const Interval *interval{};
            int k;
            {
                std::unique_lock lock{Mutex};
                Cv.wait(lock, [&] { return Stop || Exhausted || WantMoreTickets(); });
                if (Stop || Exhausted) return;
                if (!Advance()) { // tickets issue in order, so this generates exactly ticket NextTicket
                    Exhausted = true;
                    lock.unlock();
                    Cv.notify_all();
                    return;
                }
                k = NextTicket++;
                interval = &Pending.at(k); // map nodes are stable across later inserts/erases
            }
            const auto busy_start = std::chrono::steady_clock::now();
            const int n = interval->Coupled.size();
            Eigen::Array<double, 6, Eigen::Dynamic> sd1(6, n), sd2(6, n);
            for (int i = 0; i < n; ++i) {
                const auto &osc = (*Oscillators)[interval->Coupled[i]];
                sd1.col(i) = Osc::interpAt(osc, interval->T1, cursors[interval->Coupled[i]]);
                sd2.col(i) = Osc::interpAt(osc, interval->T2, cursors[interval->Coupled[i]]);
            }
            Inverses inverses;
            ConstructMassLower(sd1, sd2, interval->T1, interval->T2, interval->T1, inverses.I1);
            if (Invert(inverses.I1)) {
                ConstructMassLower(sd1, sd2, interval->T1, interval->T2, interval->T2, inverses.I2);
                inverses.Ok = Invert(inverses.I2);
            }
            StatBusyNs += (std::chrono::steady_clock::now() - busy_start).count();
            StatIntervals += 1;
            StatSumN += n;
            {
                const std::scoped_lock lock{Mutex};
                Ready.emplace(k, std::move(inverses));
            }
            Cv.notify_all();
        }
    }

    bool Fetch(double t1, double t2, int n, Eigen::MatrixXd &i1, Eigen::MatrixXd &i2) {
        const profile::Scope scope{"bubble/fetch"}; // main thread: mostly time spent waiting on workers
        std::unique_lock lock{Mutex};
        const int k = Consumed;
        Cv.wait(lock, [&] { return Stop || Ready.contains(k) || (Exhausted && k >= Generated); });
        const auto it = Pending.find(k);
        if (Stop || !Ready.contains(k) || !Ready.at(k).Ok || it->second.T1 != t1 || it->second.T2 != t2 || int(it->second.Coupled.size()) != n) {
            Stop = true; // schedule diverged, non-PD endpoint, or uncovered tail: inline from here on
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
        Cv.notify_all(); // wake workers gated on the lookahead window
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

    std::mutex Mutex;
    std::condition_variable Cv;
    std::map<int, Interval> Pending; // generated, not yet consumed
    std::map<int, Inverses> Ready;
    size_t InFlightBytes{0}; // memory held by not-yet-consumed inverses (plus generated-not-yet-computed)
    int NextTicket{0}, Generated{0}, Consumed{0};
    bool Stop{false}, Exhausted{false};
    std::vector<std::thread> Workers;

    // Worker-side stats, reported at teardown under ACOUSTIC_PROFILE
    std::atomic<uint64_t> StatBusyNs{0}, StatIntervals{0}, StatSumN{0};
};
