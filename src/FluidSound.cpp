// (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.

#include "FluidSound.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <set>
#include <stdexcept>

namespace FluidSound {

template<typename T> Solver<T>::Solver(const std::string &bubble_file, double dt, int scheme, double ts) : Dt(dt), Ts(ts) {
    std::map<int, Bubble<T>> bubbles;

    std::cout << "Reading bubble data from \"" << bubble_file << "\"" << std::endl;
    LoadBubbleFile(bubbles, bubble_file);

    MakeOscillators(bubbles);
    std::cout << "Total number of oscillators = " << Oscillators.size() << std::endl;

    if (scheme == 1) Integ = std::make_unique<CoupledDirect<T>>(dt);
    else Integ = std::make_unique<Uncoupled<T>>(dt);
}

template<typename T> T Solver<T>::Step() {
    const double time = Dt * StepIndex + Ts;

    bool lists_changed = false; // Whether TotalOsc needs rebuilding

    // Check whether any events (bubbles added or removed) occur during this timestep
    while (EventIndex < EventTimes.size() && time >= EventTimes[EventIndex]) {
        if (time < EventTimes[EventIndex + 1]) {
            lists_changed = true;
            const double time1 = EventTimes[EventIndex], time2 = EventTimes[EventIndex + 1];

            // Oscillators that have ended by time1 uncouple from the bubble cloud, but keep
            // timestepping (until they die out) to avoid discontinuities.
            int coupled_idx = 0;
            for (Oscillator<T> *osc : CoupledOsc) {
                if (time1 >= osc->EndTime) {
                    UncoupledOsc.push_back(osc);
                    continue;
                }
                CoupledOsc[coupled_idx++] = osc;
            }
            CoupledOsc.resize(coupled_idx);

            // Uncoupled oscillators that have decayed sufficiently by time1 are fully removed.
            int uncoupled_idx = 0;
            for (Oscillator<T> *osc : UncoupledOsc) {
                if (osc->IsDead()) continue;
                UncoupledOsc[uncoupled_idx++] = osc;
            }
            UncoupledOsc.resize(uncoupled_idx);

            // Oscillators starting between time1 and time2 (Oscillators is sorted by start time)
            while (OscIndex < Oscillators.size() && time >= Oscillators[OscIndex].StartTime && Oscillators[OscIndex].StartTime < time2) {
                // NOLINTNEXTLINE(misc-const-correctness) pointee cannot be const: osc is stored in CoupledOsc, a vector of non-const pointers
                Oscillator<T> *const osc = &Oscillators[OscIndex];
                if (time1 < osc->EndTime) CoupledOsc.push_back(osc);
                ++OscIndex;
            }

            // Prepare the integrator for timestepping: transfer data + refactor the mass matrix
            Integ->UpdateData(CoupledOsc, UncoupledOsc, time1, time2);
            Integ->Refactor();
        }
        ++EventIndex;
    }
    ++StepIndex;

    // The coupled/uncoupled lists only change in the event branch above, so the
    // concatenated list is rebuilt only then.
    if (lists_changed || TotalOsc.size() != CoupledOsc.size() + UncoupledOsc.size()) {
        TotalOsc.assign(CoupledOsc.begin(), CoupledOsc.end());
        TotalOsc.insert(TotalOsc.end(), UncoupledOsc.begin(), UncoupledOsc.end());
    }
    const size_t n_total = TotalOsc.size();
    if (n_total == 0) return 0.;

    Integ->Step(time);

    // Unpack the integrator's states back into the oscillators
    T total_response = 0.;
    for (size_t i = 0; i < n_total; ++i) {
        TotalOsc[i]->State(0) = Integ->States(i);
        TotalOsc[i]->State(1) = Integ->States(i + n_total);

        TotalOsc[i]->Accel = Integ->Derivs(i + n_total);
        total_response += TotalOsc[i]->Accel;
    }
    if (std::abs(total_response) > 100.) throw std::runtime_error("Instability detected!");

    return total_response;
}

template<typename T> void Solver<T>::MakeOscillators(const std::map<int, Bubble<T>> &bubbles) {
    Oscillators.clear();
    std::set<double> event_times;
    std::set<int> used_ids;

    for (const auto &[id, bubble] : bubbles) {
        if (used_ids.count(id)) continue;

        int cur_id = id;
        const Bubble<T> *cur = &bubble;

        Oscillator<T> osc;
        osc.StartTime = cur->StartTime;

        std::vector<double> solve_times;
        std::vector<T> s_radii, s_w0, s_x, s_y, s_z;
        std::vector<T> c_vals; // Precomputed damping coefficient
        std::vector<T> force_times, f_cutoffs, f_weights;

        while (true) {
            bool last = true;
            used_ids.insert(cur_id);

            solve_times.insert(solve_times.end(), cur->SolveTimes.begin(), cur->SolveTimes.end());
            s_radii.insert(s_radii.end(), cur->SolveTimes.size(), cur->Radius);
            s_w0.insert(s_w0.end(), cur->W0.begin(), cur->W0.end());
            s_x.insert(s_x.end(), cur->X.begin(), cur->X.end());
            s_y.insert(s_y.end(), cur->Y.begin(), cur->Y.end());
            s_z.insert(s_z.end(), cur->Z.begin(), cur->Z.end());

            osc.BubbleIds.push_back(cur_id);

            // ----- Bubble start event: forcing logic -----
            std::pair<T, T> force{0., 0.};
            if (cur->StartType == EventType::Split) {
                if (bubbles.at(cur->PrevIds.at(0)).Radius >= cur->Radius) force = Oscillator<T>::CzerskiJetForcing(cur->Radius);
            } else if (cur->StartType == EventType::Merge && cur->PrevIds.size() == 2) { // TODO: cleanup this code
                bool all_merge = true;
                for (const int parent_id : cur->PrevIds) all_merge = all_merge && bubbles.at(parent_id).EndType == EventType::Merge;

                if (all_merge) {
                    T r1 = bubbles.at(cur->PrevIds.at(0)).Radius;
                    T r2 = bubbles.at(cur->PrevIds.at(1)).Radius;

                    if (r1 + r2 > cur->Radius) {
                        T v1 = 4. / 3. * std::numbers::pi * r1 * r1 * r1;
                        T v2 = 4. / 3. * std::numbers::pi * r2 * r2 * r2;
                        const T vn = 4. / 3. * std::numbers::pi * cur->Radius * cur->Radius * cur->Radius;

                        const T diff = v1 + v2 - vn;
                        if (diff <= std::max(v1, v2)) {
                            if (v1 > v2) v1 -= diff;
                            else v2 -= diff;

                            r1 = std::pow(3. / 4. / std::numbers::pi * v1, 1. / 3.);
                            r2 = std::pow(3. / 4. / std::numbers::pi * v2, 1. / 3.);

                            force = Oscillator<T>::MergeForcing(cur->Radius, r1, r2);
                        }
                    } else {
                        force = Oscillator<T>::MergeForcing(cur->Radius, r1, r2);
                    }
                }
            } else if (cur->StartType == EventType::Entrain) {
                force = Oscillator<T>::CzerskiJetForcing(cur->Radius);
            }
            force_times.push_back(cur->StartTime);
            f_cutoffs.push_back(force.first);
            f_weights.push_back(force.second);

            // ----- Bubble end event: chaining logic -----
            if (cur->EndType == EventType::Merge) {
                const Bubble<T> &next = bubbles.at(cur->NextIds.at(0));
                if (LargestBubbleId(next.PrevIds, bubbles) == cur_id) {
                    last = false;
                    cur_id = cur->NextIds.at(0);
                    cur = &bubbles.at(cur_id);
                }
            } else if (cur->EndType == EventType::Split) {
                // Children in order of size
                std::multimap<T, int, std::greater<>> children;
                for (const int child_id : cur->NextIds) children.insert({bubbles.at(child_id).Radius, child_id});

                // Continue into the largest child bubble for which this is the largest parent
                for (const auto &[radius, child_id] : children) {
                    if (used_ids.count(child_id)) continue;

                    if (LargestBubbleId(bubbles.at(child_id).PrevIds, bubbles) == cur_id) {
                        last = false;
                        cur_id = child_id;
                        cur = &bubbles.at(cur_id);
                        break;
                    }
                }
            }
            if (last) break;
        }
        // End this oscillator, then decide whether to keep it
        osc.EndTime = cur->EndTime;

        if (solve_times.empty()) continue; // Not enough solve data
        if (*std::max_element(s_w0.begin(), s_w0.end()) > 2 * std::numbers::pi * 18000.) continue; // Too high frequency
        if (osc.EndTime - osc.StartTime < 3 * 2 * std::numbers::pi / s_w0[0]) continue; // Short blip

        c_vals.reserve(solve_times.size());
        for (size_t i = 0; i < solve_times.size(); ++i) c_vals.push_back(2. * Oscillator<T>::CalcBeta(s_radii[i], s_w0[i]));

        osc.SolveTimes = solve_times;
        osc.SolveData.resize(6, solve_times.size());
        osc.SolveData.row(0) = Eigen::Map<Eigen::VectorX<T>>(s_radii.data(), s_radii.size());
        osc.SolveData.row(1) = Eigen::Map<Eigen::VectorX<T>>(s_w0.data(), s_w0.size());
        osc.SolveData.row(2) = Eigen::Map<Eigen::VectorX<T>>(s_x.data(), s_x.size());
        osc.SolveData.row(3) = Eigen::Map<Eigen::VectorX<T>>(s_y.data(), s_y.size());
        osc.SolveData.row(4) = Eigen::Map<Eigen::VectorX<T>>(s_z.data(), s_z.size());
        osc.SolveData.row(5) = Eigen::Map<Eigen::VectorX<T>>(c_vals.data(), c_vals.size());

        osc.ForceData.resize(3, force_times.size());
        osc.ForceData.row(0) = Eigen::Map<Eigen::VectorX<T>>(force_times.data(), force_times.size());
        osc.ForceData.row(1) = Eigen::Map<Eigen::VectorX<T>>(f_cutoffs.data(), f_cutoffs.size());
        osc.ForceData.row(2) = Eigen::Map<Eigen::VectorX<T>>(f_weights.data(), f_weights.size());

        Oscillators.push_back(osc);
        event_times.insert(osc.StartTime);
        event_times.insert(osc.EndTime);
    }

    std::sort(Oscillators.begin(), Oscillators.end());
    EventTimes.assign(event_times.begin(), event_times.end());
}

template struct Solver<float>;
template struct Solver<double>;

} // namespace FluidSound
