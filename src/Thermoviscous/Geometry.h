#pragma once
#include "Thermoviscous.h"
namespace thermoviscous {
using UV = Eigen::Vector2d;
using Shape = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>;
using Coordinates = Eigen::Matrix<double, 3, Eigen::Dynamic, 0, 3, 9>;
struct Basis {
    Shape N, U, V;
};
int Corners(int size);
std::vector<UV> ReferenceNodes(int size, double inset = 0);
Basis BasisAt(int size, UV p, double inset = 0);
Coordinates ElementCoordinates(const Problem &p, const Element &e);
void PrepareMesh(Problem &p, const nlohmann::json &config);
} // namespace thermoviscous
