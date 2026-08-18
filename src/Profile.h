#pragma once

// Env-gated phase profiler. Set ACOUSTIC_PROFILE=1 to accumulate per-phase wall
// times and print a summary to stderr at the end of the run. Zero overhead when off
// beyond one branch per scope.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace profile {

inline bool Enabled() {
    static const bool on = std::getenv("ACOUSTIC_PROFILE") != nullptr;
    return on;
}

struct Entry {
    double Seconds{0};
    uint64_t Count{0};
};

inline std::map<std::string, Entry> &Entries() {
    static std::map<std::string, Entry> entries;
    return entries;
}

// A phase's accumulator. The map lookup costs more than the shortest phases being timed, so
// per-timestep call sites resolve theirs once into a `static profile::Entry &` and pass it to
// Scope. std::map never invalidates references, so one stays valid across later insertions.
inline Entry &Phase(const char *name) {
    if (!Enabled()) {
        static Entry unused; // Nothing reads it: a disabled scope never accumulates
        return unused;
    }
    return Entries()[name];
}

struct Scope {
    explicit Scope(Entry &entry) : E(entry) {
        if (Enabled()) Start = std::chrono::steady_clock::now();
    }
    explicit Scope(const char *name) : Scope(Phase(name)) {} // per-batch and cold call sites
    ~Scope() {
        if (!Enabled()) return;
        E.Seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - Start).count();
        E.Count += 1;
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

private:
    Entry &E;
    std::chrono::steady_clock::time_point Start;
};

inline void Report() {
    if (!Enabled()) return;
    double total = 0;
    for (const auto &[name, entry] : Entries()) total += entry.Seconds;
    fprintf(stderr, "\n==== ACOUSTIC_PROFILE (sum of instrumented phases: %.2fs) ====\n", total);
    for (const auto &[name, entry] : Entries()) {
        fprintf(stderr, "%-28s %9.3fs  (%llu calls, %.1f us/call)\n", name.c_str(), entry.Seconds, static_cast<unsigned long long>(entry.Count), entry.Count ? 1e6 * entry.Seconds / entry.Count : 0.0);
    }
}

} // namespace profile
