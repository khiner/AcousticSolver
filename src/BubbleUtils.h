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

template<typename T> struct Bubble {
    int Id{-1};
    T Radius{0}; // Effective radius

    double StartTime{-1};
    EventType StartType{}; // Entrain, merge, or split
    double EndTime{-1};
    EventType EndType{}; // Merge, split, or collapse

    std::vector<int> PrevIds; // Bubbles this one merges or splits from
    std::vector<int> NextIds; // Bubbles this one merges or splits into

    // Solve data (the bubble's own start and end times are not among these)
    std::vector<double> SolveTimes;
    std::vector<T> W0, X, Y, Z;
};

template<typename T> void LoadBubbleFile(std::map<int, Bubble<T>> &out, const std::string &filename);

template<typename T> int LargestBubbleId(const std::vector<int> &ids, const std::map<int, Bubble<T>> &bubbles);

} // namespace FluidSound
