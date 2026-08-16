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

struct Scope {
    explicit Scope(const char *name) : Name(name) {
        if (Enabled()) Start = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (!Enabled()) return;
        auto &entry = Entries()[Name];
        entry.Seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - Start).count();
        entry.Count += 1;
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

private:
    const char *Name;
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
