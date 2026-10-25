#include "Backend.h"
#include <Foundation/Foundation.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

int NumCalcAuthorMain(int, char **);
extern double ErroIterSols;

int main(int argc, char *const *argv) try {
    auto const pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    if (argc < 2) throw std::invalid_argument("Usage: NumCalcSolve PROJECT_DIR [--backend cpu|metal] [--benchmark REPEATS] [NumCalc options]");
    const auto directory = std::filesystem::absolute(argv[1]);
    if (!std::filesystem::is_regular_file(directory / "NC.inp")) throw std::invalid_argument("Project must contain NC.inp");
    std::vector<char *> arguments{argv[0]};
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--backend" && i + 1 < argc) {
            const std::string backend = argv[++i];
            if (backend != "cpu" && backend != "metal") throw std::invalid_argument("Backend must be cpu or metal");
            numcalc::Settings.Metal = backend == "metal";
        } else if (option == "--benchmark" && i + 1 < argc) {
            numcalc::Settings.BenchmarkRepeats = std::stoi(argv[++i]);
            if (numcalc::Settings.BenchmarkRepeats < 1) throw std::invalid_argument("Benchmark repeats must be positive");
        } else arguments.push_back(argv[i]);
    }
    ErroIterSols = 1e-9;
    arguments.push_back(nullptr);
    std::filesystem::current_path(directory);
    const auto start = std::chrono::steady_clock::now();
    const int status = NumCalcAuthorMain(int(arguments.size() - 1), arguments.data());
    nlohmann::json const report{{"paper", "https://doi.org/10.1016/j.enganabound.2024.01.008"}, {"upstream_commit", ACOUSTIC_NUMCALC_COMMIT}, {"backend", numcalc::Settings.Metal ? "metal" : "cpu"}, {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()}, {"frequencies", numcalc::Reports}};
    std::ofstream("native.json") << report.dump(2) << '\n';
    return status;
} catch (const std::exception &e) {
    std::cerr << "NumCalcSolve: " << e.what() << '\n';
    return 1;
}
