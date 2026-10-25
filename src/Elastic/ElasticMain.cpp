#include "Elastic.h"
#include <AudioFile.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
elastic::Json Load(const std::filesystem::path &path) {
    std::ifstream file(path);
    return elastic::Json::parse(file);
}
void Binary(const std::string &path, const double *values, size_t size) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(values), std::streamsize(size * sizeof(double)));
    if (!file) throw std::runtime_error("Cannot write " + path);
}
} // namespace
int main(int argc, char *const *argv) try {
    if (argc < 2) throw std::invalid_argument("Usage: ElasticSolve CONFIG [--output PREFIX] [--seconds N] [--prepare] [--cpu]");
    const std::filesystem::path path = argv[1];
    auto config = Load(path);
    if (config.contains("time_tolerance")) throw std::invalid_argument("time_tolerance is unsupported; integration uses fixed timesteps configured with dt");
    std::string prefix = "build/elastic/render";
    bool prepare = false, cpu = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) prefix = argv[++i];
        else if (arg == "--seconds" && i + 1 < argc) config["seconds"] = std::stod(argv[++i]);
        else if (arg == "--prepare") prepare = true;
        else if (arg == "--cpu") cpu = true;
        else throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
    const auto parent = std::filesystem::path(prefix).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    auto const mesh = Load(path.parent_path() / config.at("mesh").get<std::string>());
    const auto start = std::chrono::steady_clock::now();
    const elastic::Model model(config, mesh, cpu);
    const double preparation = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double rate = config.value("sample_rate", 48000.), duration = config.value("seconds", 1.);
    if (!std::isfinite(rate) || rate < 1 || rate > 384000 || std::floor(rate) != rate) throw std::invalid_argument("Sample rate must be an integer in 1..384000 Hz");
    elastic::Json report = {{"config", config}, {"paper", "https://doi.org/10.1016/j.jcp.2020.109632"}, {"backend", cpu ? "CPU FP64" : "Metal 4 paired float"}, {"elements", model.Elements.size()}, {"unknowns", model.Size}, {"assembled_reference_nonzeros", model.System.nonZeros()}, {"operator", cpu ? "assembled reference" : "element-local"}, {"cfl_dt", model.DtLimit}, {"cfl_real_bound", model.CflRealBound}, {"cfl_imaginary_bound", model.CflImaginaryBound}, {"prepare_seconds", preparation}};
    if (model.Viscoelastic) report["viscoelastic_paper"] = "https://doi.org/10.1016/j.camwa.2021.08.003";
    if (prepare) {
        std::cout << report.dump(2) << '\n';
        return 0;
    }
    const elastic::RunOptions options{cpu, config.value("dt", 0.)};
    const auto trace = model.Run(duration, rate, options);
    const dg::reference::Mat samples = trace.Samples, energy = trace.Energy;
    Binary(prefix + ".bin", samples.data(), samples.size());
    Binary(prefix + ".energy.bin", energy.data(), energy.size());
    Binary(prefix + ".state.bin", trace.State.data(), trace.State.size());
    double gain = 1;
    if (samples.cols()) {
        const double peak = samples.cwiseAbs().maxCoeff();
        gain = peak > 0 ? .95 / peak : 1;
        AudioFile<float> wav;
        wav.setAudioBufferSize(int(samples.cols()), int(samples.rows()));
        wav.setSampleRate(uint32_t(rate));
        wav.setBitDepth(24);
        for (int i = 0; i < samples.rows(); ++i)
            for (int ch = 0; ch < samples.cols(); ++ch) wav.samples[ch][i] = float(gain * samples(i, ch));
        if (!wav.save(prefix + ".wav")) throw std::runtime_error("Cannot write WAV");
    }
    report.update({{"audio_seconds", samples.rows() / rate}, {"render_seconds", trace.Seconds}, {"gpu_seconds", trace.GpuSeconds}, {"dt", trace.Dt}, {"backend_setup_seconds", trace.SetupSeconds}, {"steps", trace.Steps}, {"wav_gain", gain}, {"sample_count", samples.rows()}, {"first_sample_seconds", 1 / rate}, {"energy_columns", {"discrete_energy_J", "physical_energy_J", "source_power_W", "boundary_power_W", "interface_dissipation_W"}}});
    if (model.Viscoelastic) report["energy_columns"].push_back("material_dissipation_W");
    if (!(std::ofstream(prefix + ".json") << report.dump(2) << '\n')) throw std::runtime_error("Cannot write report");
    std::cout << report.dump(2) << '\n';
    return 0;
} catch (const std::exception &error) {
    std::cerr << "ElasticSolve: " << error.what() << '\n';
    return 1;
}
