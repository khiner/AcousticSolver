#include "Tube.h"
#include <AudioFile.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
void Binary(const std::string &path, const std::vector<double> &values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(double)));
    if (!out) throw std::runtime_error("Cannot write " + path);
}
double Write(const std::string &prefix, const tube::Model &model, const tube::Trace &trace) {
    for (const auto *values : {&trace.Samples, &trace.Energy, &trace.State})
        for (const double value : *values)
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite tube output");
    Binary(prefix + ".bin", trace.Samples);
    Binary(prefix + ".energy.bin", trace.Energy);
    Binary(prefix + ".state.bin", trace.State);
    double peak = 0;
    for (const double x : trace.Samples) {
        peak = std::max(peak, std::abs(x));
    }
    const double gain = peak ? .95 / peak : 1;
    AudioFile<float> wav;
    wav.setAudioBufferSize(int(model.Channels), int(model.Steps));
    wav.setSampleRate(uint32_t(model.Metadata.at("sample_rate")));
    wav.setBitDepth(24);
    for (uint32_t i = 0; i < model.Steps; ++i)
        for (uint32_t ch = 0; ch < model.Channels; ++ch) wav.samples[ch][i] = float(gain * trace.Samples[size_t(i) * model.Channels + ch]);
    if (!wav.save(prefix + ".wav")) throw std::runtime_error("Cannot write WAV");
    return gain;
}
} // namespace
int main(int argc, char *const *argv) try {
    if (argc < 2) throw std::invalid_argument("Usage: TubeSolve CONFIG [--output PREFIX] [--seconds N] [--tables]");
    std::ifstream file(argv[1]);
    auto config = nlohmann::json::parse(file);
    bool tables = false;
    std::string prefix = "build/tube/render";
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tables") tables = true;
        else if (arg == "--output" && i + 1 < argc) prefix = argv[++i];
        else if (arg == "--seconds" && i + 1 < argc) config["seconds"] = std::stod(argv[++i]);
        else throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
    if (const auto parent = std::filesystem::path(prefix).parent_path(); !parent.empty()) std::filesystem::create_directories(parent);
    const auto model = tube::Prepare(config);
    const auto trace = tube::Run(model);
    auto report = model.Metadata;
    report["backend"] = "native FP64";
    report["render_seconds"] = trace.Seconds;
    report["audio_seconds"] = model.Steps * model.Dt;
    report["wav_gain"] = Write(prefix, model, trace);
    report["paper"] = "https://doi.org/10.1121/10.0005537";
    report["openwind_commit"] = "0c88d39200f5a2af5a8c206e1ca17940b998562d";
    if (tables) {
        Binary(prefix + ".pressure_coefficients.bin", model.PressureCoefficients);
        Binary(prefix + ".flow_coefficients.bin", model.FlowCoefficients);
        Binary(prefix + ".gradient.bin", model.Gradient);
        Binary(prefix + ".input.bin", model.Input);
        Binary(prefix + ".pickups.bin", model.Pickups);
        Binary(prefix + ".positions.bin", model.Positions);
    }
    std::ofstream metadata(prefix + ".json");
    metadata << report.dump(2) << '\n';
    if (!metadata) throw std::runtime_error("Cannot write metadata");
    std::cout << report.dump(2) << '\n';
    return 0;
} catch (const std::exception &e) {
    std::cerr << "TubeSolve: " << e.what() << '\n';
    return 1;
}
