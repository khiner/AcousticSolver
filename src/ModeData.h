#pragma once

// (c) 2024 Kangrui Xue, (c) 2023 Jui-Hsien Wang. Adapted from ModalSound / openpbso (MIT) — see NOTICE.md.
// Modal displacement shapes and frequencies, as produced by modal analysis.
//
// Based on code by Jui-Hsien Wang (https://github.com/jhwang7628/openpbso)

#include <fstream>
#include <string>
#include <vector>

namespace ModalSound {

struct ModeData {
    std::vector<double> OmegaSquared;
    std::vector<std::vector<double>> Modes; // One displacement shape per eigenvalue

    void Read(const std::string &filename) {
        std::ifstream in{filename, std::ios::binary};

        // Problem size and mode count
        int n_dof, n_modes;
        in.read(reinterpret_cast<char *>(&n_dof), sizeof(int));
        in.read(reinterpret_cast<char *>(&n_modes), sizeof(int));

        OmegaSquared.resize(n_modes);
        in.read(reinterpret_cast<char *>(OmegaSquared.data()), sizeof(double) * n_modes);

        Modes.resize(n_modes);
        for (auto &mode : Modes) {
            mode.resize(n_dof);
            in.read(reinterpret_cast<char *>(mode.data()), sizeof(double) * n_dof);
        }
    }
};

} // namespace ModalSound
