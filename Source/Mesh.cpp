#include "Mesh.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <utility>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

bool ReadObjData(const std::string &filename, std::vector<double> &vs, std::vector<int> &fs) {
    std::ifstream in{filename, std::ios::binary | std::ios::ate};
    if (!in.good()) return false;
    const auto size = size_t(in.tellg());
    in.seekg(0);
    std::string data(size, '\0');
    in.read(data.data(), size);

    vs.clear();
    fs.clear();
    const char *p = data.data();
    const char *const end = p + size;
    // Advances past blanks, false once the line is finished.
    const auto more_on_line = [&] {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) ++p;
        return p < end && *p != '\n';
    };
    const auto skip_line = [&] {
        while (p < end && *p != '\n') ++p;
        if (p < end) ++p;
    };

    std::vector<int> corners;
    while (p < end) {
        const bool on_line = more_on_line();
        const bool keyword = on_line && p + 1 < end && (p[1] == ' ' || p[1] == '\t');
        if (keyword && *p == 'v') {
            ++p;
            for (int c = 0; c < 3; ++c) {
                if (!more_on_line()) return false;
                char *q{};
                const double value = std::strtod(p, &q);
                if (q == p) return false;
                vs.push_back(value);
                p = q;
            }
            // Extra components (vertex colors, a w coordinate) are dropped.
        } else if (keyword && *p == 'f') {
            ++p;
            corners.clear();
            while (more_on_line()) {
                char *q{};
                const long index = std::strtol(p, &q, 10);
                if (q == p) return false;
                // Indices are 1-based, or negative to count back from the last vertex read.
                corners.push_back(int(index < 0 ? index + long(vs.size()) / 3 : index - 1));
                p = q;
                while (p < end && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') ++p; // texture/normal indices
            }
            if (corners.size() < 3) return false;
            for (size_t c = 1; c + 1 < corners.size(); ++c) { // fan-triangulated
                fs.push_back(corners[0]);
                fs.push_back(corners[c]);
                fs.push_back(corners[c + 1]);
            }
        }
        skip_line();
    }
    return true;
}

Eigen::VectorXd MeanCurvature(const Eigen::MatrixXd &v, const Eigen::MatrixXi &f) {
    const int n_verts = int(v.rows()), n_faces = int(f.rows());
    Eigen::VectorXd curvature = Eigen::VectorXd::Zero(n_verts);
    if (n_verts == 0 || n_faces == 0) return curvature;

    // Face normals, and twice each face's area (from its projections onto the coordinate planes).
    Eigen::MatrixXd face_normals(n_faces, 3);
    Eigen::VectorXd double_areas = Eigen::VectorXd::Zero(n_faces);
    for (int i = 0; i < n_faces; ++i) {
        const Eigen::RowVector3d e1 = v.row(f(i, 1)) - v.row(f(i, 0));
        const Eigen::RowVector3d e2 = v.row(f(i, 2)) - v.row(f(i, 0));
        face_normals.row(i) = e1.cross(e2);
        const double norm = face_normals.row(i).norm();
        if (norm == 0) face_normals.row(i).setZero();
        else face_normals.row(i) /= norm;

        for (int x = 0; x < 3; ++x) {
            const int y = (x + 1) % 3;
            const double rx = v(f(i, 0), x) - v(f(i, 2), x), sx = v(f(i, 1), x) - v(f(i, 2), x);
            const double ry = v(f(i, 0), y) - v(f(i, 2), y), sy = v(f(i, 1), y) - v(f(i, 2), y);
            const double projected = rx * sy - ry * sx;
            double_areas(i) += projected * projected;
        }
        double_areas(i) = std::sqrt(double_areas(i));
    }

    // Area-weighted vertex normals.
    Eigen::MatrixXd vertex_normals = Eigen::MatrixXd::Zero(n_verts, 3);
    for (int i = 0; i < n_faces; ++i) {
        for (int c = 0; c < 3; ++c) vertex_normals.row(f(i, c)) += double_areas(i) * face_normals.row(i);
    }
    vertex_normals.rowwise().normalize();

    // Vertex-vertex adjacency, ascending and deduplicated.
    std::vector<std::vector<int>> adjacency(n_verts);
    for (int i = 0; i < n_faces; ++i) {
        for (int c = 0; c < 3; ++c) {
            const int s = f(i, c), d = f(i, (c + 1) % 3);
            adjacency[s].push_back(d);
            adjacency[d].push_back(s);
        }
    }
    for (auto &neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    static constexpr int KRing{5}; // neighborhood radius, in edges
    static constexpr size_t MinFitPoints{6}; // the quadric has 5 coefficients

    std::vector<int> neighborhood, front_facing;
    std::vector<std::pair<int, int>> queue; // (vertex, ring)
    std::vector<char> visited(n_verts, 0);
    for (int i = 0; i < n_verts; ++i) {
        // The k-ring neighborhood, breadth-first, `i` itself included.
        neighborhood.clear();
        queue.clear();
        queue.emplace_back(i, 0);
        visited[i] = 1;
        for (size_t head = 0; head < queue.size(); ++head) {
            const auto [vertex, ring] = queue[head];
            neighborhood.push_back(vertex);
            if (ring == KRing) continue;
            for (const int neighbor : adjacency[vertex]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = 1;
                    queue.emplace_back(neighbor, ring + 1);
                }
            }
        }
        for (const int vertex : neighborhood) visited[vertex] = 0;
        if (neighborhood.size() < MinFitPoints) continue;

        // Prefer the neighbors on this vertex's side of its tangent plane, if enough of them.
        const Eigen::Vector3d up = vertex_normals.row(i);
        front_facing.clear();
        for (const int vertex : neighborhood) {
            if (vertex_normals.row(vertex).dot(up) > 0.) front_facing.push_back(vertex);
        }
        if (front_facing.size() >= MinFitPoints && front_facing.size() < neighborhood.size()) neighborhood.swap(front_facing);

        // Tangent frame: the first neighbor projected onto the tangent plane, then the normal.
        const Eigen::Vector3d normal = up.normalized();
        const Eigen::Vector3d origin = v.row(i);
        const Eigen::Vector3d neighbor = v.row(adjacency[i][0]);
        const Eigen::Vector3d x_axis = (neighbor - normal * ((neighbor - origin).dot(normal)) - origin).normalized();
        const Eigen::Vector3d y_axis = normal.cross(x_axis).normalized();

        // Least-squares fit of the height field z = a u^2 + b u w + c w^2 + d u + e w.
        Eigen::MatrixXd basis(neighborhood.size(), 5), heights(neighborhood.size(), 1);
        for (size_t r = 0; r < neighborhood.size(); ++r) {
            const Eigen::Vector3d tangent = Eigen::Vector3d(v.row(neighborhood[r])) - origin;
            const double u = tangent.dot(x_axis), w = tangent.dot(y_axis);
            basis(r, 0) = u * u;
            basis(r, 1) = u * w;
            basis(r, 2) = w * w;
            basis(r, 3) = u;
            basis(r, 4) = w;
            heights(r) = tangent.dot(normal);
        }
        const Eigen::MatrixXd quadric = basis.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(heights);
        const double a = quadric(0), b = quadric(1), c = quadric(2), d = quadric(3), e = quadric(4);

        // Principal curvatures: eigenvalues of the shape operator built from the quadric's
        // first (E, F, G) and second (L, M, N) fundamental forms.
        const double E = 1. + d * d, F = d * e, G = 1. + e * e;
        const Eigen::Vector3d n = Eigen::Vector3d(-d, -e, 1.).normalized();
        const double L = 2. * a * n[2], M = b * n[2], N = 2 * c * n[2];
        Eigen::Matrix2d shape;
        shape << L * G - M * F, M * E - L * F, M * E - L * F, N * E - M * F;
        shape = shape / (E * G - F * F);

        const Eigen::Vector2d principal = -Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(shape).eigenvalues();
        curvature(i) = (principal(0) + principal(1)) / 2.;
    }
    return curvature;
}
