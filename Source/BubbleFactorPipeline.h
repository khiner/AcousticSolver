#pragma once

// Precomputes FluidSound's per-event-interval mass-matrix Cholesky factorizations on
// background worker threads, ahead of the solver consuming them.
//
// The schedule is static: oscillator trajectories are fixed at load, and event times
// quantize onto the solver's fixed step grid. The one dynamic input to Solver::step's
// event bookkeeping (is_dead pruning) touches only the uncoupled list, which the mass
// matrix excludes, so a cursor replays that bookkeeping exactly, generating intervals
// lazily as workers take tickets. Workers run the same arithmetic as the inline path, so
// results are bit-identical. Fetch verifies each interval's (t1, t2, N) against the
// schedule and permanently falls back to inline computation on any mismatch, including
// the uncovered final event, whose interval endpoint the reference reads out of bounds.

#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "FluidSound.h"

class BubbleFactorPipeline {
    using Osc = FluidSound::Oscillator<double>;
    using Llt = Eigen::LLT<Eigen::MatrixXd>;
    using Integrator = FluidSound::Coupled_Direct<double>;

    struct Interval {
        double T1, T2;
        std::vector<int> Coupled; // oscillator indices, in the solver's coupled order
    };
    struct Factors {
        Llt F1, F2;
    };

public:
    BubbleFactorPipeline(FluidSound::Solver<double> &solver, double dt, double ts)
        : Oscillators(&solver.oscillators()), Events(&solver.eventTimes()), Dt(dt), Ts(ts) {
        Integ = dynamic_cast<Integrator *>(solver.integrator());
        if (!Integ || Events->size() < 2) return;

        Integ->FactorProvider = [this](double t1, double t2, int n, Llt &f1, Llt &f2) { return Fetch(t1, t2, n, f1, f2); };
        const unsigned n_workers = std::clamp(std::thread::hardware_concurrency() / 4u, 2u, 4u);
        for (unsigned w = 0; w < n_workers; ++w) Workers.emplace_back([this] { WorkerLoop(); });
    }

    ~BubbleFactorPipeline() {
        if (Integ) Integ->FactorProvider = nullptr;
        {
            const std::scoped_lock lock{Mutex};
            Stop = true;
        }
        Cv.notify_all();
        for (auto &worker : Workers) worker.join();
    }

private:
    static constexpr int Lookahead{8}; // max intervals computed ahead of consumption

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
                Cv.wait(lock, [&] { return Stop || Exhausted || NextTicket < Consumed + Lookahead; });
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
            const int n = interval->Coupled.size();
            Eigen::Array<double, 6, Eigen::Dynamic> sd1(6, n), sd2(6, n);
            for (int i = 0; i < n; ++i) {
                const auto &osc = (*Oscillators)[interval->Coupled[i]];
                sd1.col(i) = Osc::interpAt(osc, interval->T1, cursors[interval->Coupled[i]]);
                sd2.col(i) = Osc::interpAt(osc, interval->T2, cursors[interval->Coupled[i]]);
            }
            Factors factors;
            Eigen::MatrixXd m;
            Integrator::ConstructMass(sd1, sd2, interval->T1, interval->T2, interval->T1, n, m);
            factors.F1.compute(m);
            Integrator::ConstructMass(sd1, sd2, interval->T1, interval->T2, interval->T2, n, m);
            factors.F2.compute(m);
            {
                const std::scoped_lock lock{Mutex};
                Ready.emplace(k, std::move(factors));
            }
            Cv.notify_all();
        }
    }

    bool Fetch(double t1, double t2, int n, Llt &f1, Llt &f2) {
        std::unique_lock lock{Mutex};
        const int k = Consumed;
        Cv.wait(lock, [&] { return Stop || Ready.contains(k) || (Exhausted && k >= Generated); });
        const auto it = Pending.find(k);
        if (Stop || !Ready.contains(k) || it->second.T1 != t1 || it->second.T2 != t2 || int(it->second.Coupled.size()) != n) {
            Stop = true; // schedule diverged (or uncovered tail): inline from here on
            lock.unlock();
            Cv.notify_all();
            return false;
        }
        auto node = Ready.extract(k);
        f1 = std::move(node.mapped().F1);
        f2 = std::move(node.mapped().F2);
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
    std::map<int, Factors> Ready;
    int NextTicket{0}, Generated{0}, Consumed{0};
    bool Stop{false}, Exhausted{false};
    std::vector<std::thread> Workers;
};
