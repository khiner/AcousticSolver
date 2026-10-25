#include "Geometry.h"
#include <map>
#include <stdexcept>
namespace thermoviscous {
int Corners(int size) {
    if (size == 3 || size == 6) return 3;
    if (size == 4 || size == 8 || size == 9) return 4;
    throw std::invalid_argument("Supported elements have 3/6 triangular or 4/8/9 quadrilateral nodes");
}
std::vector<UV> ReferenceNodes(int size, double inset) {
    std::vector<UV> nodes;
    if (Corners(size) == 3) nodes = {{0, 0}, {1, 0}, {0, 1}, {.5, 0}, {.5, .5}, {0, .5}};
    else nodes = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}, {0, -1}, {1, 0}, {0, 1}, {-1, 0}, {0, 0}};
    nodes.resize(size);
    for (auto &p : nodes) p = Corners(size) == 3 ? UV(UV::Constant(inset) + (1 - 3 * inset) * p) : UV((1 - inset) * p);
    return nodes;
}
Basis BasisAt(int size, UV p, double inset) {
    const double scale = Corners(size) == 3 ? 1 - 3 * inset : 1 - inset;
    p = Corners(size) == 3 ? UV((p - UV::Constant(inset)) / scale) : UV(p / scale);
    const double u = p.x(), v = p.y(), w = 1 - u - v;
    Basis b{Shape(size), Shape(size), Shape(size)};
    if (size == 3) {
        b.N << w, u, v;
        b.U << -1, 1, 0;
        b.V << -1, 0, 1;
    } else if (size == 6) {
        b.N << w * (2 * w - 1), u * (2 * u - 1), v * (2 * v - 1), 4 * w * u, 4 * u * v, 4 * v * w;
        b.U << 1 - 4 * w, 4 * u - 1, 0, 4 * (w - u), 4 * v, -4 * v;
        b.V << 1 - 4 * w, 0, 4 * v - 1, -4 * u, 4 * u, 4 * (w - v);
    } else {
        const auto nodes = ReferenceNodes(size);
        const auto poly = [&](double x, double node) {if(size==4) return UV{(1+node*x)/2,node/2};if(node==0) return UV{1-x*x,-2*x};return UV{x*(x+node)/2,x+node/2}; };
        for (int j = 0; j < size; ++j) {
            auto a = poly(u, nodes[j].x()), c = poly(v, nodes[j].y());
            b.N[j] = a.x() * c.x();
            b.U[j] = a.y() * c.x();
            b.V[j] = a.x() * c.y();
            if (size == 8) {
                const double weight = j < 4 ? -.25 : .5;
                b.N[j] += weight * (1 - u * u) * (1 - v * v);
                b.U[j] -= weight * 2 * u * (1 - v * v);
                b.V[j] -= weight * 2 * v * (1 - u * u);
            }
        }
    }
    b.U /= scale;
    b.V /= scale;
    return b;
}
Coordinates ElementCoordinates(const Problem &p, const Element &e) {
    Coordinates x(3, e.Geometry.size());
    for (size_t j = 0; j < e.Geometry.size(); ++j) x.col(j) = p.Vertices[e.Geometry[j]];
    return x;
}
void PrepareMesh(Problem &p, const nlohmann::json &config) {
    const auto order = config.value("physics_order", std::string("geometry"));
    const bool discontinuous = config.value("discontinuous", false);
    const double inset = discontinuous ? config.value("physics_inset", .1) : 0;
    if (!(inset >= 0 && inset < 1. / 3) || (discontinuous && inset == 0)) throw std::invalid_argument("Discontinuous physics_inset must lie strictly between 0 and 1/3");
    std::map<int, int> indices;
    for (auto &e : p.Elements) {
        const int corners = Corners(int(e.Geometry.size()));
        int size = int(e.Geometry.size());
        if (order == "linear") size = corners;
        else if (order == "quadratic") size = corners == 3 ? 6 : size == 8 ? 8 :
                                                                             9;
        else if (order != "geometry") throw std::invalid_argument("physics_order must be geometry, linear or quadratic; constant elements cannot resolve surface gradients");
        e.Dofs.resize(size);
        e.Inset = inset;
        for (int i : e.Geometry)
            if (i < 0 || i >= int(p.Vertices.size())) throw std::invalid_argument("Geometry node out of range");
        if (!discontinuous) {
            if (size > int(e.Geometry.size())) throw std::invalid_argument("Continuous physics order cannot exceed geometry order");
            for (int j = 0; j < size; ++j) indices.emplace(e.Geometry[j], 0);
        }
    }
    for (auto &[vertex, index] : indices) {
        index = int(p.Sources.size());
        p.Sources.push_back(p.Vertices[vertex]);
    }
    for (auto &e : p.Elements) {
        const auto nodes = ReferenceNodes(int(e.Dofs.size()), inset);
        const auto x = ElementCoordinates(p, e);
        for (size_t j = 0; j < e.Dofs.size(); ++j) {
            if (discontinuous) {
                e.Dofs[j] = int(p.Sources.size());
                p.Sources.push_back(x * BasisAt(int(e.Geometry.size()), nodes[j]).N);
            } else e.Dofs[j] = indices.at(e.Geometry[j]);
        }
    }
}
} // namespace thermoviscous
