#include "PorousScene.h"

#include "AabbTree.h"
#include "Mesh.h"
#include "Porous.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace Porous {
namespace {
using json = nlohmann::json;
using Point = std::array<double, 3>;
constexpr int Padding = 4;

Point ReadPoint(const json &value) {
    const Point p = value.get<Point>();
    for (double const x : p)
        if (!std::isfinite(x)) throw std::runtime_error("Room coordinates must be finite");
    return p;
}

// One closed, consistently oriented OBJ surface, without self intersections.
// Edge and angle-weighted vertex normals determine the sign when the closest
// point lies on a triangle boundary; global winding may be either direction.
struct Volume {
    Point Min{}, Max{};
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd Normals, VertexNormals;
    struct Edge {
        int Count{}, Orientation{}, Face{-1};
        Eigen::Vector3d Normal{Eigen::Vector3d::Zero()};
    };
    std::map<std::pair<int, int>, Edge> Edges;
    AabbTree<double> Tree;
    double Orientation{1.}, Tolerance{};

    Volume(const json &config, const std::filesystem::path &directory) {
        if (!config.contains("mesh")) {
            Min = ReadPoint(config.at("min"));
            Max = ReadPoint(config.at("max"));
            for (int a = 0; a < 3; ++a)
                if (!(Max[a] > Min[a])) throw std::runtime_error("Porous box must have positive thickness");
            return;
        }
        if (config.contains("min") || config.contains("max"))
            throw std::runtime_error("Specify either a porous box or an OBJ mesh");
        if (!ReadObj((directory / config.at("mesh").get<std::string>()).string(), V, F) || !F.rows())
            throw std::runtime_error("Cannot read porous OBJ mesh");
        if (!V.allFinite()) throw std::runtime_error("Porous mesh coordinates must be finite");
        double volume = 0.;
        Normals.resize(F.rows(), 3);
        VertexNormals = Eigen::MatrixXd::Zero(V.rows(), 3);
        std::vector<int> parent(size_t(F.rows()));
        std::iota(parent.begin(), parent.end(), 0);
        const auto component = [&](int face) {
            while (parent[size_t(face)] != face) face = parent[size_t(face)];
            return face;
        };
        for (int a = 0; a < 3; ++a) {
            Min[a] = std::numeric_limits<double>::infinity();
            Max[a] = -Min[a];
        }
        for (int t = 0; t < F.rows(); ++t) {
            for (int a = 0; a < 3; ++a) {
                int i = F(t, a), j = F(t, (a + 1) % 3);
                if (i < 0 || j < 0 || i >= V.rows() || j >= V.rows() || i == j)
                    throw std::runtime_error("Invalid porous mesh triangle");
            }
            const Eigen::Vector3d a = V.row(F(t, 0)), b = V.row(F(t, 1)), c = V.row(F(t, 2));
            const Eigen::Vector3d n = (b - a).cross(c - a);
            if (!(n.norm() > 1e-15)) throw std::runtime_error("Degenerate porous mesh triangle");
            Normals.row(t) = n.normalized();
            const Eigen::Vector3d reference = V.row(F(0, 0));
            volume += (a - reference).dot((b - reference).cross(c - reference));
            for (int corner = 0; corner < 3; ++corner) {
                const int i = F(t, corner), j = F(t, (corner + 1) % 3), k = F(t, (corner + 2) % 3);
                auto &edge = Edges[std::minmax(i, j)];
                ++edge.Count;
                edge.Orientation += i < j ? 1 : -1;
                edge.Normal += Normals.row(t).transpose();
                if (edge.Face >= 0) parent[size_t(component(t))] = component(edge.Face);
                else edge.Face = t;
                const Eigen::Vector3d u = V.row(j) - V.row(i), v = V.row(k) - V.row(i);
                VertexNormals.row(i) += std::atan2(u.cross(v).norm(), u.dot(v)) * Normals.row(t);
                for (int axis = 0; axis < 3; ++axis) {
                    Min[axis] = std::min(Min[axis], V(i, axis));
                    Max[axis] = std::max(Max[axis], V(i, axis));
                }
            }
        }
        for (const auto &[key, edge] : Edges)
            if (edge.Count != 2 || edge.Orientation != 0)
                throw std::runtime_error("Porous mesh must be closed and consistently oriented");
        for (int t = 1; t < F.rows(); ++t)
            if (component(t) != component(0))
                throw std::runtime_error("Each porous OBJ must contain one connected surface; use separate volumes");
        if (std::abs(volume) < 1e-15) throw std::runtime_error("Porous mesh encloses zero signed volume");
        Orientation = volume < 0. ? -1. : 1.;
        Tolerance = 1e-12 * std::max({Max[0] - Min[0], Max[1] - Min[1], Max[2] - Min[2]});
        Tree.Init(V, F);
    }

    bool Contains(const Point &p) const {
        for (int a = 0; a < 3; ++a)
            if (p[a] < Min[a] || p[a] >= Max[a]) return false;
        if (!F.rows()) return true;
        const Eigen::RowVector3d point{p[0], p[1], p[2]};
        Eigen::RowVector3d nearest;
        int face;
        Tree.ClosestPoint(point, face, nearest);
        if (face < 0) return false;
        Eigen::RowVector3d normal = Normals.row(face);
        for (int corner = 0; corner < 3; ++corner) {
            const int i = F(face, corner);
            if ((nearest - V.row(i)).norm() <= Tolerance)
                return Orientation * (point - nearest).dot(VertexNormals.row(i)) <= 0.;
        }
        for (int corner = 0; corner < 3; ++corner) {
            const int i = F(face, corner), j = F(face, (corner + 1) % 3);
            const Eigen::Vector3d edge = V.row(j) - V.row(i), from = nearest - V.row(i);
            if (edge.cross(from).norm() <= Tolerance * edge.norm()) {
                normal = Edges.at(std::minmax(i, j)).Normal;
                break;
            }
        }
        return Orientation * (point - nearest).dot(normal) <= 0.;
    }
};
} // namespace

std::array<double, 3> Position(const RoomScene &scene, int node, const std::array<double, 3> &origin) {
    return {origin[0] + scene.H * (node / (scene.Ny * scene.Nz)), origin[1] + scene.H * UnfoldedY(scene, node), origin[2] + scene.H * (node % scene.Nz)};
}

void AddVolumes(RoomScene &scene, const json &config, const std::filesystem::path &directory, const Point &origin) {
    if (!config.contains("porous_volumes") || config.at("porous_volumes").empty()) return;
    if (!scene.Fcc) throw std::runtime_error("Porous volumes require the FCC room scheme");
    std::vector<Material> materials;
    for (const auto &m : config.at("porous_materials")) {
        const auto band = m.value("fit_hz", std::array<double, 2>{20., 4000.});
        if (band[1] >= .5 / scene.Ts) throw std::runtime_error("Porous fit band exceeds the time-step Nyquist frequency");
        materials.push_back(PrepareMaterial(m.at("flow_resistivity"), band[0], band[1]));
        const auto &fit = materials.back();
        std::printf("[porous] material %zu: %.1f..%.1f Hz, fit error %.4f%%; %.2f grid spacings/air wavelength at upper fit frequency (dispersion excluded)\n", materials.size() - 1, band[0], band[1], 100. * fit.MaxRelativeError, scene.C / (band[1] * scene.H));
    }
    std::vector<int> labels(size_t(scene.NumNodes()), -1);
    for (const auto &v : config.at("porous_volumes")) {
        const int material = v.at("material");
        if (material < 0 || material >= int(materials.size()))
            throw std::runtime_error("Porous volume references an undefined material");
        const Volume volume(v, directory);
        if (config.contains("room")) {
            const Point size = ReadPoint(config.at("room").at("size"));
            for (int a = 0; a < 3; ++a)
                if (volume.Min[a] < 0. || volume.Max[a] > size[a] + 1e-12)
                    throw std::runtime_error("Porous volume extends outside the native room");
        }
        int count = 0;
        for (int i = 0; i < scene.NumNodes(); ++i) {
            if (!IsInterior(scene, i) || !volume.Contains(Position(scene, i, origin))) continue;
            if (labels[i] != -1) throw std::runtime_error("Porous volumes overlap on the simulation grid");
            labels[i] = material;
            ++count;
        }
        if (!count) throw std::runtime_error("Porous volume contains no FCC nodes; refine the grid");
    }
    for (int const receiver : scene.OutIxyz)
        if (labels[receiver] >= 0) throw std::runtime_error("Room receivers must be in air");
    scene.Porous = Prepare(scene, labels, materials);
}

RoomScene MakeRoom(const json &config, const std::filesystem::path &directory) {
    const auto &room = config.at("room");
    RoomScene scene;
    scene.Fcc = true;
    scene.PressureOutput = true;
    if (config.value("scheme", std::string("fcc")) != "fcc")
        throw std::runtime_error("Native porous room scenes use the FCC scheme");
    scene.H = room.at("h");
    scene.C = room.value("c", 343.);
    scene.Rho = room.value("rho", 1.2);
    scene.L = room.value("courant", .9);
    const double seconds = room.value("seconds", .2);
    if (!(scene.H > 0.) || !(scene.C > 0.) || !(scene.Rho > 0.) || !(scene.L > 0. && scene.L <= 1.) ||
        !(seconds > 0.) || !std::isfinite(scene.H + scene.C + scene.Rho + scene.L + seconds))
        throw std::runtime_error("Invalid native room grid, medium, duration, or Courant number");
    scene.L2 = scene.L * scene.L;
    scene.Ts = scene.L * scene.H / scene.C;
    scene.Srate = 1. / scene.Ts;
    if (seconds / scene.Ts > std::numeric_limits<int>::max()) throw std::runtime_error("Room duration exceeds index range");
    scene.Nt = int(std::ceil(seconds / scene.Ts));
    const Point size = ReadPoint(room.at("size"));
    std::array<int, 3> cells;
    for (int a = 0; a < 3; ++a) {
        if (!(size[a] / scene.H >= 4.) || size[a] / scene.H > 1000000.)
            throw std::runtime_error("Native room extent must span at least four grid cells");
        cells[a] = int(std::ceil(size[a] / scene.H));
    }
    scene.Nx = cells[0] + 2 * Padding;
    const int ny_full = ((cells[1] + 2 * Padding + 1) / 2) * 2;
    scene.Ny = ny_full / 2 + 1;
    scene.Nz = cells[2] + 2 * Padding;
    if (int64_t(scene.Nx) * scene.Ny * scene.Nz > std::numeric_limits<int>::max())
        throw std::runtime_error("Room exceeds 32-bit node indexing");
    const Point origin{(.5 - Padding) * scene.H, (.5 - Padding) * scene.H, (.5 - Padding) * scene.H};
    const auto inside = [&](int node) {
        const auto p = Position(scene, node, origin);
        for (int a = 0; a < 3; ++a)
            if (p[a] < 0. || p[a] >= cells[a] * scene.H) return false;
        return true;
    };
    for (int i = 0; i < scene.NumNodes(); ++i) {
        if (!IsInterior(scene, i)) continue;
        const auto neighbours = FccNeighbours(scene, i);
        const bool inside_room = inside(i);
        uint16_t adjacency = 0;
        for (int b = 0; b < 12; ++b)
            if (inside_room == inside(neighbours[b])) adjacency |= uint16_t(1 << b);
        if (adjacency == 4095) continue;
        scene.BnIxyz.push_back(i);
        scene.AdjBn.push_back(adjacency);
        scene.MatBn.push_back(-1);
        scene.SafBn.push_back(1.);
    }
    const auto snap = [&](const json &value) {
        const Point p = ReadPoint(value);
        for (int a = 0; a < 3; ++a)
            if (p[a] < 0. || p[a] >= size[a]) throw std::runtime_error("Source/receiver lies outside room");
        std::array<int, 3> q;
        for (int a = 0; a < 3; ++a) q[a] = int(std::lround((p[a] - origin[a]) / scene.H));
        int nearest = -1;
        double distance = std::numeric_limits<double>::infinity();
        for (int x = q[0] - 1; x <= q[0] + 1; ++x)
            for (int y = q[1] - 1; y <= q[1] + 1; ++y)
                for (int z = q[2] - 1; z <= q[2] + 1; ++z) {
                    if ((x + y + z) & 1) continue;
                    if (x < 1 || x >= scene.Nx - 1 || y < 1 || y >= ny_full - 1 || z < 1 || z >= scene.Nz - 1) continue;
                    const int i = (x * scene.Ny + std::min(y, ny_full - 1 - y)) * scene.Nz + z;
                    if (!inside(i)) continue;
                    const auto s = Position(scene, i, origin);
                    double d = 0.;
                    for (int a = 0; a < 3; ++a) d += (p[a] - s[a]) * (p[a] - s[a]);
                    if (d < distance) {
                        distance = d;
                        nearest = i;
                    }
                }
        if (nearest < 0) throw std::runtime_error("Cannot place source/receiver on FCC grid");
        return nearest;
    };
    const auto &source = config.at("source");
    scene.InIxyz.push_back(snap(source.at("position")));
    const double peak = source.value("peak_hz", 800.);
    if (!(peak > 0. && peak < .5 * scene.Srate)) throw std::runtime_error("Invalid source peak frequency");
    scene.InSigs.resize(scene.Nt);
    const double sigma = 1. / (2. * std::numbers::pi * peak);
    for (int n = 0; n < scene.Nt; ++n) {
        const double q = (n * scene.Ts - 6. * sigma) / sigma;
        scene.InSigs[n] = -q * std::exp(-.5 * q * q);
    }
    scene.CornersPerReceiver = 1;
    for (const auto &receiver : config.at("receivers")) {
        scene.OutIxyz.push_back(snap(receiver));
        scene.OutAlpha.push_back(1.);
    }
    scene.NumReceivers = scene.NumOutputs();
    if (!scene.NumReceivers) throw std::runtime_error("Room needs at least one receiver");
    scene.Output = config.value("output", std::string("porous-room"));
    AddVolumes(scene, config, directory, origin);
    return scene;
}
} // namespace Porous
