// (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.
//
// References:
//   [Langlois et al. 2016] Toward Animating Water with Complex Acoustic Bubbles

#include "Oscillator.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

namespace FluidSound {

namespace {
constexpr double RhoWater{998.}; // Density of water
constexpr double Gamma{1.4}; // Gas heat capacity ratio
constexpr double Sigma{0.0726}; // Surface tension
constexpr double Mu{8.9e-4}; // Dynamic viscosity of water
constexpr double GTh{1.6e6}; // Thermal damping constant
constexpr double G{1.0}; // TODO: ?
constexpr double Cf{1497}; // Speed of sound in water
constexpr double Atm{101325}; // Atmospheric pressure

constexpr double MaxCutoff{0.0006};

std::default_random_engine ForcingRnd;
std::uniform_real_distribution<double> Frac{0.4, 0.8};
} // namespace

// Eq. 14 from [Langlois et al. 2016]
template<typename T> std::pair<T, T> Oscillator<T>::CzerskiJetForcing(T radius) {
    const T eta = 0.95; // TODO

    const T cutoff = std::min(MaxCutoff, 0.5 / (3. / radius)); // 1/2 minnaert period

    const T pressure_in0 = (Atm + 2. * Sigma / radius);
    const T weight = -9. * Gamma * Sigma * eta * pressure_in0 * std::sqrt(1. + eta * eta) / (4. * RhoWater * radius * radius * radius);

    const T mass = (RhoWater / (4. * std::numbers::pi * radius));

    return {cutoff, weight / mass};
}

// Eq. 15 from [Langlois et al. 2016]
template<typename T> std::pair<T, T> Oscillator<T>::MergeForcing(T radius, T r1, T r2) {
    const T frac = Frac(ForcingRnd);
    const T factor = std::pow(2. * Sigma * r1 * r2 / (RhoWater * (r1 + r2)), 0.25);

    T cutoff = std::min(MaxCutoff, 0.5 / (3. / radius)); // 1/2 minnaert period
    const T tmp = std::pow(frac * std::min(r1, r2) / 2. / factor, 2); // TODO: cleanup
    cutoff = std::min(cutoff, tmp);

    const T pressure_in0 = (Atm + 2. * Sigma / radius);
    const T weight = 6. * Sigma * Gamma * pressure_in0 / (RhoWater * radius * radius * radius);

    const T mass = (RhoWater / (4. * std::numbers::pi * radius));

    return {cutoff, weight / mass};
}

template<typename T> T Oscillator<T>::CalcBeta(T radius, T w0) {
    const T dr = w0 * radius / Cf;
    const T dvis = 4 * Mu / (RhoWater * w0 * radius * radius);
    const T phi = 16. * GTh * G / (9 * (Gamma - 1) * (Gamma - 1) * w0 / 2. / std::numbers::pi);
    const T dth = 2 * (std::sqrt(phi - 3) - (3 * Gamma - 1) / (3 * (Gamma - 1))) / (phi - 4);

    const T dtotal = dr + dvis + dth;

    return w0 * dtotal / std::sqrt(dtotal * dtotal + 4);
}

template struct Oscillator<float>;
template struct Oscillator<double>;

} // namespace FluidSound
