// (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.

#include "BubbleUtils.h"

#include <fstream>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace FluidSound {

namespace {
// Returns false at end of file.
template<typename T> bool ParseBubble(Bubble<T> &bub, std::ifstream &in) {
    std::string line;
    std::istringstream is;

    int id;
    double time;
    T freq_hz, x, y, z;
    T pressure = 101450.; // Default for a line that omits it

    // Line 1: 'Bub <unique bubble ID> <radius>'
    std::getline(in, line);
    if (line.empty()) return false;

    is = std::istringstream(line.substr(4));
    is >> bub.Id;
    is >> bub.Radius;

    // Line 2: '  Start: <event type> <start time> <previous bubble ID(s)>'
    std::getline(in, line);
    switch (line[9]) {
        case 'N': bub.StartType = EventType::Entrain; break;
        case 'M': bub.StartType = EventType::Merge; break;
        case 'S': bub.StartType = EventType::Split; break;
        default: throw std::runtime_error("Invalid bubble start event type");
    }
    is = std::istringstream(line.substr(11));
    is >> bub.StartTime;
    while (is >> id) bub.PrevIds.push_back(id);

    // Line 3-n: '  <time> <freqHz> <x> <y> <z> <pressure inside bubble (optional)>'
    std::getline(in, line);
    while (line[2] != 'E' && line[2] != 'B' && !in.eof()) {
        is = std::istringstream(line);
        is >> time >> freq_hz >> x >> y >> z >> pressure;
        std::getline(in, line);

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
        is = std::istringstream(line.substr(9));
        is >> bub.EndTime;
        while (is >> id) bub.NextIds.push_back(id);

        if (bub.EndType == EventType::Merge && bub.NextIds.size() != 1) throw std::runtime_error(std::to_string(id) + " did not merge to one bubble");
        if (bub.EndType == EventType::Split && bub.NextIds.size() < 2) throw std::runtime_error(std::to_string(id) + " split to less than 2 bubbles");
    } else if (line[2] == 'B') { // ...however, if end info is not present, use default values
        bub.EndTime = bub.SolveTimes.empty() ? bub.StartTime + 0.001 : bub.SolveTimes.back();
        bub.EndType = EventType::Collapse;
    }
    return true;
}
} // namespace

template<typename T> void LoadBubbleFile(std::map<int, Bubble<T>> &out, const std::string &filename) {
    out.clear();
    std::ifstream in{filename};
    while (in.good()) {
        Bubble<T> bub;
        if (ParseBubble(bub, in)) out.insert({bub.Id, std::move(bub)});
    }
}

template<typename T> int LargestBubbleId(const std::vector<int> &ids, const std::map<int, Bubble<T>> &bubbles) {
    T max_radius = 0.;
    int max_id = 0;
    for (const int id : ids) {
        if (bubbles.at(id).Radius > max_radius) {
            max_radius = bubbles.at(id).Radius;
            max_id = id;
        }
    }
    return max_id;
}

template void LoadBubbleFile<float>(std::map<int, Bubble<float>> &, const std::string &);
template void LoadBubbleFile<double>(std::map<int, Bubble<double>> &, const std::string &);
template int LargestBubbleId<float>(const std::vector<int> &, const std::map<int, Bubble<float>> &);
template int LargestBubbleId<double>(const std::vector<int> &, const std::map<int, Bubble<double>> &);

} // namespace FluidSound
