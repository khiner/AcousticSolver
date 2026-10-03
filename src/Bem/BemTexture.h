#pragma once
#include "Bem.h"
#include <memory>

namespace bem {
using Texture = Eigen::VectorXf;
using UVs = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
UVs HeightUVs(const Problem &);
class HeightfieldMap {
public:
    HeightfieldMap(const UVs &, int width, int height);
    ~HeightfieldMap();
    Eigen::VectorXd Sample(const Texture &);
    Texture Transpose(const Eigen::VectorXd &);

private:
    struct Impl;
    std::unique_ptr<Impl> Data;
};
} // namespace bem
