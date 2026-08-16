#pragma once

// Chunked parallel-for over [0, n) using libdispatch. Used only for loops whose
// iterations are independent and write disjoint (or identical) values, so results
// are bit-identical to the serial loop.

#include <algorithm>
#include <cstddef>
#include <dispatch/dispatch.h>
#include <thread>

// Calls fn(begin, end) on contiguous chunks covering [0, n). Runs serially when the
// loop is too small to be worth the fan-out.
template<typename F> void ParallelChunks(size_t n, size_t min_grain, F &&fn) {
    if (n == 0) return;

    static const size_t n_workers = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t n_chunks = std::min(n_workers * 4, std::max<size_t>(1, n / min_grain));
    if (n_chunks <= 1) {
        fn(size_t{0}, n);
        return;
    }
    struct Ctx {
        F *Fn;
        size_t Chunk, N;
    };
    Ctx ctx{&fn, (n + n_chunks - 1) / n_chunks, n};
    dispatch_apply_f(n_chunks, DISPATCH_APPLY_AUTO, &ctx, [](void *p, size_t c) {
        auto &[fn, chunk, n_total] = *static_cast<Ctx *>(p);
        const size_t begin = c * chunk;
        if (begin < n_total) (*fn)(begin, std::min(begin + chunk, n_total));
    });
}

// Calls fn(i) for each i in [0, n), in parallel chunks.
template<typename F> void ParallelFor(size_t n, size_t min_grain, F &&fn) {
    ParallelChunks(n, min_grain, [&fn](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) fn(i);
    });
}
