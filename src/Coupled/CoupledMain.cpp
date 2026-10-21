#include "Coupled.h"
#include "Profile.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void Binary(const std::string &path, const std::vector<double> &values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(double)));
    if (!file) throw std::runtime_error("Cannot write " + path);
}
double Wav(const std::string &path, const coupled::Tables &t, const std::vector<double> &samples) {
    const uint16_t channels = uint16_t(t.Receivers.size() - t.Plates.size());
    const uint64_t length = uint64_t(t.Steps) * channels * sizeof(float);
    if (length > UINT32_MAX - 36) throw std::runtime_error("WAV exceeds RIFF size limit");
    double peak = 0;
    for (uint32_t i = 0; i < t.Steps; ++i)
        for (uint16_t ch = 0; ch < channels; ++ch) peak = std::max(peak, std::abs(samples[size_t(i) * t.Receivers.size() + ch]));
    const double gain = peak ? .95 / peak : 1;
    std::ofstream file(path, std::ios::binary);
    auto const word = [&](auto value) { file.write(reinterpret_cast<const char *>(&value), sizeof(value)); };
    file.write("RIFF", 4);
    word(uint32_t(36 + length));
    file.write("WAVEfmt ", 8);
    word(uint32_t(16));
    word(uint16_t(3));
    word(channels);
    word(uint32_t(t.SampleRate));
    word(uint32_t(t.SampleRate) * channels * 4);
    word(uint16_t(channels * 4));
    word(uint16_t(32));
    file.write("data", 4);
    word(uint32_t(length));
    for (uint32_t i = 0; i < t.Steps; ++i)
        for (uint16_t ch = 0; ch < channels; ++ch) word(float(gain * samples[size_t(i) * t.Receivers.size() + ch]));
    if (!file) throw std::runtime_error("Cannot write " + path);
    return gain;
}
double RelativeError(const std::vector<double> &a, const std::vector<double> &b, size_t stride = 1, size_t channel = 0) {
    double diff = 0, norm = 0;
    for (size_t i = channel; i < a.size(); i += stride) {
        diff += (a[i] - b[i]) * (a[i] - b[i]);
        norm += b[i] * b[i];
    }
    return std::sqrt(diff / std::max(norm, 1e-300));
}
} // namespace
int main(int argc, char *const *argv) try {
    if (argc < 2) {
        std::cerr << "Usage: CoupledSolve config.json [--cpu|--compare] [--output PREFIX] [--seconds N]\n";
        return 1;
    }
    std::ifstream file(argv[1]);
    auto config = nlohmann::json::parse(file);
    bool cpu = false, compare = false;
    std::string prefix = "build/coupled/render";
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cpu") cpu = true;
        else if (arg == "--compare") compare = true;
        else if (arg == "--output" && i + 1 < argc) prefix = argv[++i];
        else if (arg == "--seconds" && i + 1 < argc) config["seconds"] = std::stod(argv[++i]);
        else throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
    if (cpu && compare) throw std::invalid_argument("Choose --cpu or --compare");
    if (!std::filesystem::path(prefix).parent_path().empty()) std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
    coupled::Reference reference(config);
    const auto &t = reference.Data;
    auto const trace = cpu ? reference.Run() : coupled::RunMetal(t);
    auto report = t.Metadata;
    report["backend"] = cpu ? t.Metadata.at("reference_backend").get<std::string>() : "Metal float-pair";
    report["render_seconds"] = trace.Seconds;
    report["audio_seconds"] = t.Steps / t.SampleRate;
    report["metal_buffer_bytes"] = trace.BufferBytes;
    report["wav_gain"] = Wav(prefix + ".wav", t, trace.Samples);
    Binary(prefix + ".bin", trace.Samples);
    Binary(prefix + ".air.bin", trace.Air);
    Binary(prefix + ".panels.bin", trace.Displacements);
    bool passed = true;
    if (compare) {
        auto const upstream = reference.Run();
        report["reference_seconds"] = upstream.Seconds;
        auto const check = [&](double error) { passed &= error < 1e-4; return error; };
        report["air_relative_error"] = check(RelativeError(trace.Air, upstream.Air));
        report["panel_relative_error"] = check(RelativeError(trace.Displacements, upstream.Displacements));
        for (size_t ch = 0; ch < t.Receivers.size(); ++ch)
            report["channel_relative_errors"].push_back(check(RelativeError(trace.Samples, upstream.Samples, t.Receivers.size(), ch)));
        report["comparison_passed"] = passed;
        const auto *const suffix = report["reference_backend"] == "NESS FP64" ? ".upstream" : ".reference";
        report["reference_output"] = prefix + suffix;
        Binary(prefix + suffix + ".bin", upstream.Samples);
        report["reference_wav_gain"] = Wav(prefix + suffix + ".wav", t, upstream.Samples);
    }
    std::ofstream metadata(prefix + ".json");
    metadata << report.dump(2) << '\n';
    if (!metadata) throw std::runtime_error("Cannot write render metadata");
    std::cout << report.dump(2) << '\n';
    profile::Report();
    return passed ? 0 : 2;
} catch (const std::exception &e) {
    std::cerr << "CoupledSolve: " << e.what() << '\n';
    return 1;
}
