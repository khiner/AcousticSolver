#pragma once

// (c) 2024 Kangrui Xue. Adapted from FluidSound (MIT) — see NOTICE.md.
// A single physical bubble of a fluid simulation, and reading a bubble file from disk.

#include <map>
#include <string>
#include <vector>

namespace FluidSound {

enum class EventType {
    Entrain,
    Merge,
    Split,
    Collapse,
};

struct Bubble {
    int Id{-1};
    double Radius{0}; // Effective radius

    double StartTime{-1};
    EventType StartType{}; // Entrain, merge, or split
    double EndTime{-1};
    EventType EndType{}; // Merge, split, or collapse

    std::vector<int> PrevIds; // Bubbles this one merges or splits from
    std::vector<int> NextIds; // Bubbles this one merges or splits into

    // Solve data (the bubble's own start and end times are not among these)
    std::vector<double> SolveTimes;
    std::vector<double> W0, X, Y, Z;
};

void LoadBubbleFile(std::map<int, Bubble> &out, const std::string &filename);

int LargestBubbleId(const std::vector<int> &ids, const std::map<int, Bubble> &bubbles);

} // namespace FluidSound
