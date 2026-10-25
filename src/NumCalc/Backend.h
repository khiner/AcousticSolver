#pragma once
#include "NC_TypeDefinition.h"
#include "json.hpp"
#include <memory>
#include <vector>

namespace numcalc {
struct Matrix {
    int Rows{}, Columns{};
    const int *Offsets{}, *Indices{};
    const Complex *Values{};
    size_t Entries() const { return Offsets ? size_t(Offsets[Rows]) : size_t(Rows) * Columns; }
};
struct Level {
    Matrix T, D, S;
};
struct Operator {
    Matrix Near;
    std::vector<Level> Levels;
    void Apply(const Complex *x, Complex *y) const;
    size_t Bytes() const;
};
class Gpu {
public:
    explicit Gpu(const Operator &);
    ~Gpu();
    void Apply(const Complex *, Complex *);

private:
    struct Impl;
    std::unique_ptr<Impl> Data;
};
struct Options {
    bool Metal{};
    int BenchmarkRepeats{};
    double ResidualLimit{1e-7};
};
extern Options Settings;
extern nlohmann::json Reports;
} // namespace numcalc
