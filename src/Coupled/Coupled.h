#pragma once

#include "CoupledParams.h"
#include <array>
#include <cstdint>
#include <json.hpp>
#include <memory>
#include <string>
#include <vector>

namespace coupled {
using Matrix = CoupledMatrix;
struct Plate {
    uint32_t Size{}, Stride{}, Iterations{}, LowerSlice{};
    bool Nonlinear{}, Adjoint{};
    double WScale{}, FScale{}, InvH2{}, Bf{}, Alpha{}, H{};
    Matrix B, C, K, IJ, I, J;
    uint32_t Derivatives{}, Mask{}, Inverse{}, MassInverse{};
    std::vector<double> AirMask;
};
struct Strike {
    uint32_t Plate{}, Index{}, Start{}, Length{};
    double Amplitude{};
};
struct Receiver {
    int Plate{-1};
    uint32_t Index{};
};
struct Tables {
    uint32_t Nx{}, Ny{}, Nz{}, Steps{}, MaxIterations{64};
    double SampleRate{}, PressureScale{}, Q{}, Gamma{}, Lambda{}, AirDensity{}, Tolerance{1e-9};
    bool Reflecting{}, Lossless{};
    std::vector<uint32_t> Indices;
    std::vector<double> Values, AirCoefficients;
    std::vector<uint8_t> AirIndex;
    std::vector<Plate> Plates;
    std::vector<Strike> Strikes;
    std::vector<Receiver> Receivers;
    nlohmann::json Metadata;
    uint32_t Append(const double *values, size_t count);
    Matrix Store(uint32_t rows, const int *row, const int *column, const double *values);
};
struct Trace {
    std::vector<double> Samples, Air, Displacements;
    double Seconds{};
    size_t BufferBytes{};
};

// NESS supplies the baseline; explicit rectangular edge conditions use native operators and an FP64 direct solve.
class Reference {
public:
    explicit Reference(const nlohmann::json &config);
    ~Reference();
    Reference(const Reference &) = delete;
    Reference &operator=(const Reference &) = delete;
    Tables Data;
    Trace Run();

private:
    struct Impl;
    std::unique_ptr<Impl> State;
};
Trace RunMetal(const Tables &tables);
} // namespace coupled
