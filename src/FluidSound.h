#pragma once

// (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.
// High-level manager for bubble-based water sound synthesis.
//
// Based on code from Timothy Langlois and Ryan Aronson. Thanks to Zhehao Li for reviewing!

#include "BubbleUtils.h"
#include "Integrators.h"

#include <memory>

namespace FluidSound {

template<typename T> struct Solver {
    // Reads bubble data from `bubble_file` and initializes the oscillators.
    // `scheme` selects the coupling scheme (0 uncoupled, 1 coupled), `ts` the simulation
    // start time.
    Solver(const std::string &bubble_file, double dt, int scheme, double ts = 0.);

    // Timesteps the oscillator vibrations, returning their summed volume acceleration.
    T Step();

    std::vector<Oscillator<T>> Oscillators; // All oscillators, sorted by start time
    std::vector<double> EventTimes; // Sorted event times, i.e. when to refactor the mass matrix
    // The integrator, exposed so a precomputed-factor provider can be installed
    // (see BubbleFactorPipeline.h).
    std::unique_ptr<Integrator<T>> Integ;

private:
    // Chains bubbles together into oscillators.
    void MakeOscillators(const std::map<int, Bubble<T>> &bubbles);

    double Dt{0}; // Timestep size
    double Ts{0}; // Simulation start time
    int StepIndex{0};

    std::vector<Oscillator<T> *> CoupledOsc, UncoupledOsc;
    std::vector<Oscillator<T> *> TotalOsc; // Coupled + uncoupled, rebuilt only when they change

    int OscIndex{0}, EventIndex{0}; // Cursors into Oscillators and EventTimes
};

} // namespace FluidSound
