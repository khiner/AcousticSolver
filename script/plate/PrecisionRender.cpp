// Independent convergence reference for scaled nonlinear plate trajectories.
#include "PrecisionModel.h"

#ifndef PLATE_DECIMAL_DIGITS
#define PLATE_DECIMAL_DIGITS 64
#endif
using Real = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<PLATE_DECIMAL_DIGITS>>;

int main(int argc, char *const *argv) {
    try {
        if (argc != 4) throw std::runtime_error("PrecisionRender config.json prefix max_steps (0 means full)");
        Json config;
        std::ifstream(argv[1]) >> config;
        // Only scaled nonlinear fixtures with exact coefficients are supported.
        for (auto it = config.begin(); it != config.end(); ++it)
            if (it.key() != "sample_rate" && it.key() != "seconds" && it.key() != "cutoff_hz" && it.key() != "kappa" && it.key() != "ratio" && it.key() != "sigma0" && it.key() != "sigma1" && it.key() != "lambda" && it.key() != "strikes" && it.key() != "pickups" && it.key() != "repeat_count" && it.key() != "initial_amplitude" && it.key() != "norm_type") throw std::runtime_error("Unsupported diagnostic configuration key: " + it.key());
        plate::Config settings;
        settings.SampleRate = config.at("sample_rate");
        settings.Kappa = config.at("kappa");
        settings.Ratio = config.at("ratio");
        settings.CutoffHz = config.at("cutoff_hz");
        settings.Sigma0 = config.at("sigma0");
        settings.Sigma1 = config.at("sigma1");
        settings.Lambda = config.at("lambda");
        settings.NormType = config.value("norm_type", 1);
        const auto p = plate::Prepare(settings);
        const Model<Real> model(p);
        const size_t modes = p.Mx * p.My;
        const size_t cycle = size_t(config.at("seconds").get<double>() * settings.SampleRate);
        const size_t total = cycle * config.value("repeat_count", size_t(1));
        const size_t limit = std::stoull(argv[3]);
        const size_t steps = limit ? std::min(limit, total) : total;
        if (!cycle || !steps || settings.SampleRate != 44100) throw std::runtime_error("Invalid diagnostic duration or sample rate");
        State<Real> state{Vec<Real>(modes), Vec<Real>(modes), Real(0)};
        state.q[0] = state.previous[0] = config.value("initial_amplitude", 0.);
        state.psi = sqrt(2 * model.potential(state.q, false).value + model.epsilon);
        auto const basis = [&](const Json &position) {
            const Real x(position.at(0).get<double>()), y(position.at(1).get<double>());
            Vec<Real> out(modes);
            for (size_t i = 0; i < modes; ++i)
                if (p.Active[i]) out[i] = 2 * sin(model.kx[i] * x) * sin(model.ky[i] * y);
            return out;
        };
        std::vector<Vec<Real>> pickups, excitation;
        for (const auto &point : config.at("pickups")) pickups.push_back(basis(point));
        for (const auto &strike : config.at("strikes")) excitation.push_back(basis(strike.at("position")));
        const std::string prefix = argv[2];
        std::ofstream wave(prefix + ".bin", std::ios::binary), metrics(prefix + ".diagnostics.bin", std::ios::binary), states(prefix + ".states.bin", std::ios::binary);
        wave.exceptions(std::ios::badbit | std::ios::failbit);
        metrics.exceptions(std::ios::badbit | std::ios::failbit);
        states.exceptions(std::ios::badbit | std::ios::failbit);
        auto const write = [](std::ofstream &out, const Real &value) {
            const double v = double(value);
            if (!std::isfinite(v)) throw std::runtime_error("Nonfinite reference output");
            out.write(reinterpret_cast<const char *>(&v), sizeof(v));
        };
        const Real pi = acos(Real(-1));
        const auto began = std::chrono::steady_clock::now();
        double maxUpdateResidual = 0;
        for (size_t n = 0; n < steps; ++n) {
            Vec<Real> force(modes), half(modes), delta(modes);
            const Real time = Real(n % cycle) * model.dt;
            size_t j = 0;
            for (const auto &strike : config.at("strikes")) {
                const Real start(strike.value("start", 0.)), duration(strike.at("duration").get<double>());
                if (duration <= 0 || strike.value("type", 2) < 1) throw std::runtime_error("Invalid strike");
                if (time >= start && time <= start + duration) {
                    const Real amplitude = Real(.5) * Real(strike.at("amplitude").get<double>()) * (1 - cos(Real(strike.value("type", 2)) * pi * (time - start) / duration));
                    for (size_t i = 0; i < modes; ++i) force[i] += amplitude * excitation[j][i];
                }
                ++j;
            }
            Real norm = 0, energy = Real(.5) * model.kappa * model.kappa * state.psi * state.psi;
            for (size_t i = 0; i < modes; ++i) {
                half[i] = Real(.5) * (state.q[i] + state.previous[i]);
                delta[i] = (state.q[i] - state.previous[i]) / model.dt;
                norm += settings.NormType == 1 ? Real(abs(delta[i])) : Real(delta[i] * delta[i]);
                energy += Real(.5) * (delta[i] * delta[i] + model.omega[i] * state.q[i] * state.previous[i]);
            }
            const auto potential = model.potential(state.q);
            const Real halfPotential = model.potential(half, false).value;
            const Real drift = state.psi - sqrt(2 * halfPotential + model.epsilon);
            auto g = model.g(potential);
            const Real control = -Real(settings.Lambda) * drift / (norm + model.epsilon);
            for (size_t i = 0; i < modes; ++i) g[i] += control * (settings.NormType == 1 ? Real((delta[i] > 0) - (delta[i] < 0)) : delta[i]);
            const auto before = state;
            model.advance(state, g, force);
            // Independent algebraic update check during excitation, after it,
            // and at each subsequent repeated strike, including regulation.
            if (n % cycle == 50 || n % cycle == 256) {
                const auto original = model.original(before, g, force);
                Real error = 0, scale = 0;
                for (size_t i = 0; i < modes; ++i) {
                    error += (original.q[i] - state.q[i]) * (original.q[i] - state.q[i]);
                    scale += state.q[i] * state.q[i];
                }
                if (scale > 0) maxUpdateResidual = std::max(maxUpdateResidual, double(sqrt(error / scale)));
            }
            Real inputPower = 0, lossPower = 0;
            for (size_t i = 0; i < modes; ++i) {
                const Real velocity = (state.q[i] - before.previous[i]) / (2 * model.dt);
                inputPower += velocity * force[i];
                lossPower += 2 * model.sigma[i] * velocity * velocity;
                write(states, state.q[i]);
            }
            for (const auto &pickup : pickups) {
                Real sample = 0;
                for (size_t i = 0; i < modes; ++i) sample += pickup[i] * state.q[i];
                write(wave, sample);
            }
            for (const Real &value : {energy, inputPower, lossPower, before.psi, halfPotential, drift}) write(metrics, value);
            if ((n + 1) % 4410 == 0) {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
                std::cerr << n + 1 << '/' << steps << " samples; " << elapsed << " seconds\n";
            }
        }
        wave.close();
        metrics.close();
        states.close();
        Vec<double> q, previous;
        for (const auto &v : state.q) q.push_back(double(v));
        for (const auto &v : state.previous) previous.push_back(double(v));
        Json metadata = {{"config", config}, {"sample_count", steps}, {"full_render", steps == total}, {"mx", p.Mx}, {"my", p.My}, {"nx", p.Nx}, {"ny", p.Ny}, {"bits", mpfr_get_prec(Real(0).backend().data())}, {"mpfr_version", MPFR_VERSION_STRING}, {"original_update_relative_residual", maxUpdateResidual}, {"simulation_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count()}, {"final_state", {{"q", q}, {"previous", previous}, {"psi", double(state.psi)}}}, {"semantics", "Binary64 configuration values promoted exactly; fixed native grid/masks; coefficients, initialization, point bases, sampled forcing and dynamics evaluated at MPFR precision. Output rounded to binary64 only when written. States and pickups after update; energy, psi, half potential and drift before update."}};
        std::ofstream(prefix + ".json") << metadata.dump(2) << '\n';
        std::cout << metadata["bits"] << " bits: " << steps << " samples in " << metadata["simulation_seconds"] << " seconds\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
