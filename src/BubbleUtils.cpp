// (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.

#include "BubbleUtils.h"

#include <charconv>
#include <fstream>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace FluidSound {

namespace {
// Line reader over the whole file (bubble files run to tens of MB). from_chars rounds
// identically to the stream extractors, so parsed values are bit-identical.
struct LineCursor {
    const char *Pos, *End;
    bool Eof{false}; // mirrors the stream's eofbit: set when a read runs out of data

    // std::getline emulation: consumes up to (and including) the newline.
    std::string_view GetLine() {
        if (Pos >= End) {
            Eof = true;
            return {};
        }
        const char *begin = Pos;
        while (Pos < End && *Pos != '\n') ++Pos;
        std::string_view line{begin, size_t(Pos - begin)};
        if (Pos < End) ++Pos; // consume '\n'
        else Eof = true;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        return line;
    }
};

// `is >> value` emulation on a line: skip whitespace, then parse. Returns false (leaving
// `value` untouched) once the line runs out, like a stream entering the fail state.
template<typename T> bool Extract(std::string_view &line, T &value) {
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    line.remove_prefix(i);
    if (line.empty()) return false;
    const auto [ptr, ec] = std::from_chars(line.data(), line.data() + line.size(), value);
    if (ec != std::errc{}) return false;
    line.remove_prefix(ptr - line.data());
    return true;
}

// Returns false at end of file.
bool ParseBubble(Bubble &bub, LineCursor &in) {
    // Line 1: 'Bub <unique bubble ID> <radius>'
    std::string_view line = in.GetLine();
    if (line.empty()) return false;

    line.remove_prefix(4);
    Extract(line, bub.Id);
    Extract(line, bub.Radius);

    // Line 2: '  Start: <event type> <start time> <previous bubble ID(s)>'
    line = in.GetLine();
    switch (line[9]) {
        case 'N': bub.StartType = EventType::Entrain; break;
        case 'M': bub.StartType = EventType::Merge; break;
        case 'S': bub.StartType = EventType::Split; break;
        default: throw std::runtime_error("Invalid bubble start event type");
    }
    line.remove_prefix(11);
    Extract(line, bub.StartTime);
    for (int id; Extract(line, id);) bub.PrevIds.push_back(id);

    // Line 3-n: '  <time> <freqHz> <x> <y> <z> <pressure inside bubble (optional, unused)>'
    line = in.GetLine();
    while (line[2] != 'E' && line[2] != 'B' && !in.Eof) {
        double time, freq_hz, x, y, z;
        std::string_view fields = line;
        Extract(fields, time);
        Extract(fields, freq_hz);
        Extract(fields, x);
        Extract(fields, y);
        Extract(fields, z);
        line = in.GetLine();

        bub.SolveTimes.push_back(time);
        bub.W0.push_back(2 * std::numbers::pi * freq_hz);
        bub.X.push_back(x);
        bub.Y.push_back(y);
        bub.Z.push_back(z);
    }

    // Line n: '  End: <event type> <end time> <next bubble ID(s)>'...
    if (line[2] == 'E') {
        switch (line[7]) {
            case 'M': bub.EndType = EventType::Merge; break;
            case 'S': bub.EndType = EventType::Split; break;
            case 'C': bub.EndType = EventType::Collapse; break;
            default: throw std::runtime_error("Invalid bubble end event type");
        }
        line.remove_prefix(9);
        Extract(line, bub.EndTime);
        int id = 0;
        while (Extract(line, id)) bub.NextIds.push_back(id);

        if (bub.EndType == EventType::Merge && bub.NextIds.size() != 1) throw std::runtime_error(std::to_string(id) + " did not merge to one bubble");
        if (bub.EndType == EventType::Split && bub.NextIds.size() < 2) throw std::runtime_error(std::to_string(id) + " split to less than 2 bubbles");
    } else if (line[2] == 'B') { // ...however, if end info is not present, use default values
        bub.EndTime = bub.SolveTimes.empty() ? bub.StartTime + 0.001 : bub.SolveTimes.back();
        bub.EndType = EventType::Collapse;
    }
    return true;
}
} // namespace

void LoadBubbleFile(std::map<int, Bubble> &out, const std::string &filename) {
    out.clear();
    std::ifstream const in{filename, std::ios::binary};
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string contents = ss.str();

    LineCursor cursor{contents.data(), contents.data() + contents.size()};
    while (!cursor.Eof) {
        Bubble bub;
        if (ParseBubble(bub, cursor)) out.insert({bub.Id, std::move(bub)});
    }
}

int LargestBubbleId(const std::vector<int> &ids, const std::map<int, Bubble> &bubbles) {
    double max_radius = 0.;
    int max_id = 0;
    for (const int id : ids) {
        if (bubbles.at(id).Radius > max_radius) {
            max_radius = bubbles.at(id).Radius;
            max_id = id;
        }
    }
    return max_id;
}

} // namespace FluidSound
