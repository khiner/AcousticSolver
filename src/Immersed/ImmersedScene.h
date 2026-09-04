#pragma once

#include "ImmersedGpu.h"

#include <string>

namespace immersed {

struct Scene {
    Grid GridSpec;
    std::vector<Point> Sources;
    std::vector<Point> Receivers;
    std::vector<Patch> Patches;
    std::vector<float> SourceSamples;
    int Steps{0};
    int InterpolationOrder{5};
    double SampleRate{0.};
    std::string Output;
    std::string SourceFile;
};

Scene LoadScene(const std::string &config_file, double seconds = 0., const std::string &output = {});
std::vector<float> RenderScene(const Scene &scene);
void RunScene(const std::string &config_file, double seconds = 0., const std::string &output = {});

} // namespace immersed
