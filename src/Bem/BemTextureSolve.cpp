#include "BemIO.h"
#include "BemTexture.h"
#include <fstream>
#include <iostream>

int main(int argc, char *const *argv) {
    try {
        if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--sample-only")) throw std::invalid_argument("Usage: BemTextureSolve design.json result.json [--sample-only]");
        nlohmann::json input;
        std::ifstream(argv[1]) >> input;
        auto problem = bem::ReadProblem(input.at("problem"));
        const auto &design = input.at("design");
        const auto values = design.at("initial_pixels").get<std::vector<float>>();
        const bem::Texture pixels = Eigen::Map<const bem::Texture>(values.data(), values.size());
        bem::HeightfieldMap map(bem::HeightUVs(problem), design.at("width"), design.at("height"));
        const auto heights = map.Sample(pixels);
        for (size_t i = 0; i < problem.HeightVertices.size(); ++i) problem.Vertices(problem.HeightVertices[i], 1) += heights[i];
        nlohmann::json output;
        if (argc == 3) {
            bem::SolverOptions options;
            if (input.contains("solver")) {
                const auto &solver = input.at("solver");
                options.SolveTolerance = solver.value("solve_tolerance", options.SolveTolerance);
                options.CompressionTolerance = solver.value("compression_tolerance", options.CompressionTolerance);
            }
            const auto result = bem::SolveMetal(problem, options);
            const bem::Texture gradient = map.Transpose(-result.HeightGradient);
            output = bem::WriteResult(result, false);
            output["loss"] = 1. - result.Diffusion;
            output["gradient"] = std::vector<float>(gradient.data(), gradient.data() + gradient.size());
            output["solver"] = {{"compression_tolerance", options.CompressionTolerance}, {"solve_tolerance", options.SolveTolerance}};
        } else {
            output["vertices"] = nlohmann::json::array();
            for (int i = 0; i < problem.Vertices.rows(); ++i) output["vertices"].push_back({problem.Vertices(i, 0), problem.Vertices(i, 1), problem.Vertices(i, 2)});
        }
        output["frequency_hz"] = problem.Frequency;
        std::ofstream file(argv[2]);
        file << output.dump() << '\n';
        if (!file) throw std::runtime_error("Cannot save BEM texture evaluation");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
