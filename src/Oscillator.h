#pragma once

// (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.
// One bubble oscillator, plus its forcing and damping models.

#include <Eigen/Dense>

#include <utility>
#include <vector>

namespace FluidSound {

// A single oscillator, v''(t) + 2 beta v'(t) + w0^2 v(t) = p(t) / m.
// Where a Bubble is a physical bubble, an Oscillator is the mathematical abstraction the
// integrator consumes: one chain of bubbles followed through its merges and splits.
struct Oscillator {
    std::vector<int> BubbleIds; // By increasing start time

    double StartTime{-1}, EndTime{-1};

    Eigen::Vector2d State{0., 0.}; // Current volume displacement and volume velocity, [v v']
    double Accel{0}; // Current volume acceleration, v''

    std::vector<double> SolveTimes; // Times [t(0) ... t(N)] of SolveData's columns
    // For indices (0, ..., N), SolveData is:
    //   [[ r(0)  ... r(N)  ]
    //    [ w0(0) ... w0(N) ]
    //    [ x(0)  ... x(N)  ]
    //    [ y(0)  ... y(N)  ]
    //    [ z(0)  ... z(N)  ]
    //    [ 2b(0) ... 2b(N) ]]
    Eigen::Array<double, 6, Eigen::Dynamic> SolveData;

    // For indices (0, ..., F), ForceData is:
    //   [[ force_time(0) ... force_time(F) ]
    //    [ cutoff(0)     ... cutoff(F)     ]
    //    [ weight(0)     ... weight(F)     ]]
    // Every forcing model has the form F(t) = (t < cutoff) * weight * t * t, where t is
    // relative to the force start time: t = time - force_time.
    Eigen::Array<double, 3, Eigen::Dynamic> ForceData;

    // Linearly interpolated solve data at `time`, walking `idx` to the bracketing column.
    // The cursor is explicit so the factor-precompute workers can carry their own
    // (see BubbleFactorPipeline.h).
    static Eigen::Array<double, 6, 1> InterpAt(const Oscillator &osc, double time, int &idx) {
        if (time >= osc.SolveTimes.back()) return osc.SolveData.col(osc.SolveTimes.size() - 1);
        if (time <= osc.SolveTimes[0]) return osc.SolveData.col(0);

        while (time < osc.SolveTimes[idx]) --idx;
        while (idx + 1 < int(osc.SolveTimes.size()) && time > osc.SolveTimes[idx + 1]) ++idx;

        const double alpha = (time - osc.SolveTimes[idx]) / (osc.SolveTimes[idx + 1] - osc.SolveTimes[idx]);
        return (1. - alpha) * osc.SolveData.col(idx) + alpha * osc.SolveData.col(idx + 1);
    }
    Eigen::Array<double, 6, 1> Interp(double time) { return InterpAt(*this, time, Idx); }

    bool IsDead() const { return State.norm() < 1e-10; }

    bool operator<(const Oscillator &osc) const { return StartTime < osc.StartTime; }

    // Neck collapse forcing model from Czerski/Deane [2008; 2010], as a (cutoff, weight) pair.
    static std::pair<double, double> CzerskiJetForcing(double radius);
    // Neck expansion forcing model from [Czerski 2011] for a bubble of `radius` born from
    // parents of radii `r1` and `r2`, as a (cutoff, weight) pair.
    static std::pair<double, double> MergeForcing(double radius, double r1, double r2);
    // Damping via radiative, viscous, and thermal effects.
    static double CalcBeta(double radius, double w0);

private:
    int Idx{0};
};

} // namespace FluidSound
