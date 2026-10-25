#pragma once
#include "Thermoviscous.h"

namespace thermoviscous {
using Points = Eigen::Matrix<double, 3, Eigen::Dynamic>;
struct FastData {
    Points Sources, Targets, Normals;
    Sparse Coefficients, CorrectionH, CorrectionG;
    Vector FreeTerm;
};
Vector Fmm(const FastData &data, Complex k, const Vector &coefficients, bool doubleLayer, double tolerance);
std::unique_ptr<AcousticOperator> MakeAcoustic(FastData data, Complex k, const Settings &settings);
} // namespace thermoviscous
