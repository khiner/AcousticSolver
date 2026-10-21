#pragma once

#include "Coupled.h"
#include <Eigen/SparseCore>

class AirboxIndexed;

namespace coupled {
using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
struct BoundaryOperators {
    Sparse Bending, Laplacian, Airy;
    Eigen::VectorXd InverseMass;
};
BoundaryOperators BuildBoundaryOperators(int nx, int ny, double h, double nu, const std::string &boundary);
void PrepareBoundary(Tables &tables, Plate &plate, const nlohmann::json &panel);
Trace RunBoundaryReference(const Tables &tables, AirboxIndexed &air);
Sparse ReadMatrix(const Tables &tables, Matrix matrix, int columns);
Sparse NonlinearOperator(const Tables &tables, const Plate &plate, const Eigen::VectorXd &displacement);
} // namespace coupled
