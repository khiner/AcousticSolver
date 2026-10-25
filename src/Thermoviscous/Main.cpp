#include "Thermoviscous.h"
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) try {
    if (argc < 2 || argc > 4) throw std::invalid_argument("Usage: ThermoviscousSolve SCENE.json [OUTPUT.json] [--operators|--operator-probes]");
    const std::string mode = argc == 4 ? argv[3] : "";
    if (!mode.empty() && mode != "--operators" && mode != "--operator-probes") throw std::invalid_argument("Unknown argument");
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("Cannot read scene " + std::string(argv[1]));
    const auto p = thermoviscous::Read(nlohmann::json::parse(input));
    const auto a = thermoviscous::Assemble(p);
    const auto s = thermoviscous::Solve(p, a);
    const std::filesystem::path output = argc >= 3 ? argv[2] : "build/thermoviscous/result.json";
    if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
    nlohmann::json report{{"paper", "https://doi.org/10.1142/S2591728523500159"}, {"frequency", p.Frequency}, {"nodes", p.Sources.size()}, {"backend", "FP64 CPU with Accelerate"}, {"time_convention", "exp(-i omega t)"}, {"assembly_seconds", a.AssemblySeconds}, {"solve_seconds", s.SolveSeconds}, {"iterations", s.Iterations}, {"inner_iterations", s.InnerIterations}, {"relative_residual", s.RelativeResidual}, {"divergence_residual", s.BoundaryResidual}};
    const auto values = [](const thermoviscous::Vector &v) {
        auto j = nlohmann::json::array();
        for (auto x : v) j.push_back({x.real(), x.imag()});
        return j;
    };
    report["method"] = p.Options.Method;
    report["pressure_seconds"] = s.PressureSeconds;
    report["recovery_seconds"] = s.RecoverySeconds;
    report["acoustic_backend"] = p.Options.AcousticBackend;
    report["acoustic_bytes"] = a.Acoustic ? a.Acoustic->Bytes() : size_t(a.Ha.size() + a.Ga.size()) * sizeof(thermoviscous::Complex);
    report["collocation_points"] = nlohmann::json::array();
    for (const auto &v : p.Sources) report["collocation_points"].push_back({v.x(), v.y(), v.z()});
    report["exterior"] = p.Options.Exterior;
    report["normal_convention"] = "into fluid";
    report["quadrature"] = p.Options.Quadrature;
    report["loss_neighborhood"] = p.Options.IncidentLosses ? "incident" : "distance";
    report["acoustic_pressure"] = values(s.Pa);
    report["thermal_pressure"] = values(s.Ph);
    report["receiver_pressure"] = values(s.Pressure);
    report["acoustic_derivative"] = values(s.Dpa);
    report["thermal_derivative"] = values(s.Dph);
    report["viscous_velocity"] = values(s.Vv);
    report["viscous_derivative"] = values(s.Dvv);
    if (mode == "--operator-probes") {
        const int n = int(p.Sources.size());
        for (int probe = 0; probe < 3; ++probe) {
            thermoviscous::Vector x(n);
            for (int i = 0; i < n; ++i) x[i] = {std::sin((i + 1) * (probe + 1) * 1.234), std::cos((i + 1) * (probe + 2) * .789)};
            report["operator_probes"].push_back({{"H", values(a.ApplyH(x))}, {"G", values(a.ApplyG(x))}});
        }
    }
    if (mode == "--operators") {
        const auto write = [&](const char *name, const thermoviscous::Matrix &m) {
            const auto file = output.parent_path() / (std::string(name) + ".bin");
            std::ofstream out(file, std::ios::binary);
            out.write(reinterpret_cast<const char *>(m.data()), std::streamsize(m.size() * sizeof(thermoviscous::Complex)));
            if (!out) throw std::runtime_error("Cannot write operator " + file.string());
            report["operators"][name] = {{"rows", m.rows()}, {"cols", m.cols()}};
        };
        if (a.Acoustic) {
            const int n = int(a.Gh.rows());
            if (n > 2048) throw std::invalid_argument("Operator export is limited to 2048 nodes");
            thermoviscous::Matrix h(n, n), g(n, n);
            for (int j = 0; j < n; ++j) {
                auto e = thermoviscous::Vector::Unit(n, j).eval();
                h.col(j) = a.ApplyH(e);
                g.col(j) = a.ApplyG(e);
            }
            write("Ha", h);
            write("Ga", g);
        } else {
            write("Ha", a.Ha);
            write("Ga", a.Ga);
        }
        write("Hh", a.Hh);
        write("Gh", a.Gh);
        write("Hv", a.Hv);
        write("Gv", a.Gv);
        write("Dc", a.Dc);
        write("Nd", a.Nd);
    }
    const auto temporary = output.string() + ".tmp";
    std::ofstream file(temporary);
    file << report.dump(2) << '\n';
    file.close();
    if (!file) throw std::runtime_error("Cannot write output");
    std::filesystem::rename(temporary, output);
    std::cout << p.Sources.size() << " nodes, assembly " << a.AssemblySeconds << " s, solve " << s.SolveSeconds << " s, residual " << s.RelativeResidual << "\n"
              << output << '\n';
    return 0;
} catch (const std::exception &e) {
    std::cerr << "ThermoviscousSolve: " << e.what() << '\n';
    return 1;
}
