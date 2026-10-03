#include "BemIO.h"
#include <fstream>
#include <iostream>

int main(int argc, char *const *argv) {
    try {
        if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--reference")) {
            std::cerr << "Usage: BemSolve problem.json result.json [--reference]\n";
            return 2;
        }
        std::ifstream input(argv[1]);
        nlohmann::json j;
        input >> j;
        const auto p = bem::ReadProblem(j.contains("problem") ? j.at("problem") : j);
        bem::SolverOptions options;
        if (j.contains("solver")) {
            const auto &s = j.at("solver");
            const auto kind = s.value("preconditioner", std::string("two_level"));
            if (kind == "none") options.Precondition = bem::Preconditioner::None;
            else if (kind == "block_jacobi") options.Precondition = bem::Preconditioner::BlockJacobi;
            else if (kind != "two_level") throw std::invalid_argument("Unknown BEM preconditioner");
            options.LeafSize = s.value("leaf_size", options.LeafSize);
            options.Restart = s.value("restart", options.Restart);
        }
        const auto r = argc == 4 ? bem::SolveReference(p) : bem::SolveMetal(p, options);
        auto output = bem::WriteResult(r);
        output["method"] = argc == 4 ? "FP64 CPU" : "FP32 compressed Metal operators and derivatives; FP64 host ACA certification, GMRES orthogonalization and incident field";
        output["frequency_hz"] = p.Frequency;
        output["triangles"] = p.Faces.rows();
        std::ofstream file(argv[2]);
        file << output.dump(2) << '\n';
        if (!file) throw std::runtime_error("Cannot write BEM result");
        std::cout << "diffusion " << r.Diffusion << ", " << r.HeightGradient.size() << " height derivatives\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
