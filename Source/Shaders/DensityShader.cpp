// Ported from WaveBlender (c) 2024 Kangrui Xue (DensityShader logic in Shaders.h) — Metal port.
// Implements the Density ("auxiliary beta") shader: reads a per-frame beta field text file.
//
// The parser reads the whole file once and walks it with strtof, reproducing istream
// getline/extraction semantics exactly, quirks included: a row is emitted per getline
// attempt while the previous line was non-empty and the stream good — so a file ending
// in a newline emits one trailing all-zeros row, and a file with no trailing newline
// ends after its last data row.

#include <cstdlib>
#include <iomanip>
#include <sstream>

#include "Shaders.h"

void Density::ReadDensity() {
    auto &base = Obj;
    std::vector<Eigen::Vector3<REAL>> pos;
    std::vector<int> betas;

    const int frame = (base.Step * base.Dt + base.Ts) * Fps + 1;
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << frame;
    const auto filename = BetaDir + "betaField_6x40x6_" + ss.str() + ".txt";

    std::ifstream in_file{filename, std::ios::binary};
    const bool open = in_file.is_open();
    std::string text;
    if (open) text.assign(std::istreambuf_iterator<char>{in_file}, std::istreambuf_iterator<char>{});

    // getline emulation over `text`: returns the line, updates `good` like a stream
    // (eofbit when no delimiter is found, failbit when nothing can be read).
    size_t cursor = 0;
    bool good = open;
    std::string line;
    const auto next_line = [&] {
        if (!good && cursor >= text.size()) { // failed stream: line untouched by a failed getline
            line.clear();
            return;
        }
        if (cursor >= text.size()) { // nothing left: failbit
            line.clear();
            good = false;
            return;
        }
        const size_t nl = text.find('\n', cursor);
        if (nl == std::string::npos) { // no delimiter: eofbit
            line.assign(text, cursor);
            cursor = text.size();
            good = false;
        } else {
            line.assign(text, cursor, nl - cursor);
            cursor = nl + 1;
        }
    };

    if (open) next_line(); // Skip first line
    while (!line.empty() && good) {
        next_line();

        // istringstream{line} >> density >> px >> py >> pz, with 0 on failed extraction
        const char *p = line.c_str();
        REAL values[4]{};
        for (auto &value : values) {
            char *next = nullptr;
            const REAL v = std::strtof(p, &next);
            if (next == p) break; // extraction failed: this and all later values stay 0
            value = v;
            p = next;
        }
        pos.emplace_back(values[1], values[2], values[3]);
        betas.push_back(int(100 * values[0]));
    }

    base.V2 = Eigen::MatrixX<REAL>::Zero(pos.size(), 3);
    base.F = Eigen::MatrixXi::Zero(betas.size(), 3);
    for (int r = 0; r < int(pos.size()); ++r) {
        base.V2.row(r) = pos[r];
        base.F(r, 0) = betas[r]; // Hack to encode density in cell face
    }
    base.Changed = open;
}
