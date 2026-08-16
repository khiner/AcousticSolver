#pragma once

// Axis-aligned bounding box hierarchy over a triangle mesh, for closest-point queries.
// The split rule (median barycenter rank along the box's longest axis) and the child-visit
// order mirror the reference implementation, so queries resolve to the triangle it picks.

#include <Eigen/Core>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

template<typename Scalar> struct AabbTree {
    using Vertices = Eigen::MatrixX<Scalar>;
    using RowVector3 = Eigen::Matrix<Scalar, 1, 3>;

    // Queries must pass the same mesh this was built over.
    void Init(const Vertices &v, const Eigen::MatrixXi &f) {
        Nodes.clear();
        if (v.rows() == 0 || f.rows() == 0) return;

        // Each triangle's rank among the barycenters along each axis. Splitting on rank
        // rather than value keeps the halves balanced when barycenters share a coordinate.
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> bc(f.rows(), 3);
        for (int i = 0; i < f.rows(); ++i) {
            bc.row(i) = (v.row(f(i, 0)) + v.row(f(i, 1)) + v.row(f(i, 2))) / Scalar(3);
        }
        Eigen::MatrixXi ranks(f.rows(), 3);
        std::vector<int> order(f.rows());
        for (int d = 0; d < 3; ++d) {
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return bc(a, d) < bc(b, d); });
            for (int i = 0; i < int(order.size()); ++i) ranks(order[i], d) = i;
        }

        std::vector<int> elements(f.rows());
        std::iota(elements.begin(), elements.end(), 0);
        Nodes.reserve(2 * f.rows());
        Build(v, f, ranks, elements);
    }

    // Index of the triangle closest to `p`, and the closest point on it.
    void ClosestPoint(const Vertices &v, const Eigen::MatrixXi &f, const RowVector3 &p, int &i_out, RowVector3 &c_out) const {
        i_out = -1;
        c_out.setZero();
        if (Nodes.empty()) return;

        Scalar sqr_d = std::numeric_limits<Scalar>::infinity();
        Search(0, v, f, p, sqr_d, i_out, c_out);
    }

private:
    struct Node {
        Scalar Min[3], Max[3];
        int Left{-1}, Right{-1};
        int Primitive{-1}; // triangle index, -1 for interior nodes
    };

    std::vector<Node> Nodes; // Nodes[0] is the root

    // Returns the new node's index.
    int Build(const Vertices &v, const Eigen::MatrixXi &f, const Eigen::MatrixXi &ranks, const std::vector<int> &elements) {
        const int node = int(Nodes.size());
        Nodes.emplace_back();

        Scalar lo[3], hi[3];
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::numeric_limits<Scalar>::max();
            hi[d] = std::numeric_limits<Scalar>::lowest();
        }
        for (const int e : elements) {
            for (int c = 0; c < 3; ++c) {
                const int vertex = f(e, c);
                for (int d = 0; d < 3; ++d) {
                    lo[d] = std::min(lo[d], v(vertex, d));
                    hi[d] = std::max(hi[d], v(vertex, d));
                }
            }
        }
        std::copy_n(lo, 3, Nodes[node].Min);
        std::copy_n(hi, 3, Nodes[node].Max);

        if (elements.size() == 1) {
            Nodes[node].Primitive = elements[0];
            return node;
        }

        int axis = 0;
        for (int d = 1; d < 3; ++d) {
            if (hi[d] - lo[d] > hi[axis] - lo[axis]) axis = d;
        }
        std::vector<int> keys(elements.size());
        for (size_t i = 0; i < elements.size(); ++i) keys[i] = ranks(elements[i], axis);
        std::vector<int> nth = keys;
        const size_t mid = (nth.size() - 1) / 2;
        std::nth_element(nth.begin(), nth.begin() + mid, nth.end());
        const int median = nth[mid];

        // Ranks are distinct, so this always splits into two non-empty halves.
        std::vector<int> left, right;
        left.reserve((elements.size() + 1) / 2);
        right.reserve(elements.size() / 2);
        for (size_t i = 0; i < elements.size(); ++i) (keys[i] <= median ? left : right).push_back(elements[i]);

        const int l = Build(v, f, ranks, left);
        const int r = Build(v, f, ranks, right);
        Nodes[node].Left = l;
        Nodes[node].Right = r;
        return node;
    }

    void Search(int node, const Vertices &v, const Eigen::MatrixXi &f, const RowVector3 &p, Scalar &sqr_d, int &i, RowVector3 &c) const {
        const Node &n = Nodes[node];
        if (n.Primitive >= 0) {
            const RowVector3 candidate = ClosestPointOnTriangle(p, v.row(f(n.Primitive, 0)), v.row(f(n.Primitive, 1)), v.row(f(n.Primitive, 2)));
            const Scalar d = (p - candidate).squaredNorm();
            if (d < sqr_d) {
                sqr_d = d;
                i = n.Primitive;
                c = candidate;
            }
            return;
        }
        // Descend into any child whose box contains the query point, then into the
        // remaining ones nearest-first, skipping those that cannot beat the current best.
        bool looked_left = false, looked_right = false;
        if (Contains(Nodes[n.Left], p)) {
            Search(n.Left, v, f, p, sqr_d, i, c);
            looked_left = true;
        }
        if (Contains(Nodes[n.Right], p)) {
            Search(n.Right, v, f, p, sqr_d, i, c);
            looked_right = true;
        }
        const Scalar left_d = ExteriorSquaredDistance(Nodes[n.Left], p);
        const Scalar right_d = ExteriorSquaredDistance(Nodes[n.Right], p);
        if (left_d < right_d) {
            if (!looked_left && left_d < sqr_d) Search(n.Left, v, f, p, sqr_d, i, c);
            if (!looked_right && right_d < sqr_d) Search(n.Right, v, f, p, sqr_d, i, c);
        } else {
            if (!looked_right && right_d < sqr_d) Search(n.Right, v, f, p, sqr_d, i, c);
            if (!looked_left && left_d < sqr_d) Search(n.Left, v, f, p, sqr_d, i, c);
        }
    }

    static bool Contains(const Node &n, const RowVector3 &p) {
        for (int d = 0; d < 3; ++d) {
            if (p[d] < n.Min[d] || p[d] > n.Max[d]) return false;
        }
        return true;
    }

    static Scalar ExteriorSquaredDistance(const Node &n, const RowVector3 &p) {
        Scalar sqr_d{0};
        for (int d = 0; d < 3; ++d) {
            if (n.Min[d] > p[d]) {
                const Scalar aux = n.Min[d] - p[d];
                sqr_d += aux * aux;
            } else if (p[d] > n.Max[d]) {
                const Scalar aux = p[d] - n.Max[d];
                sqr_d += aux * aux;
            }
        }
        return sqr_d;
    }

    // Closest point on triangle (a, b, c) to p. Real-Time Collision Detection, Ericson, ch. 5.
    static RowVector3 ClosestPointOnTriangle(const RowVector3 &p, const RowVector3 &a, const RowVector3 &b, const RowVector3 &c) {
        const RowVector3 ab = b - a, ac = c - a, ap = p - a;
        const Scalar d1 = ab.dot(ap), d2 = ac.dot(ap);
        if (d1 <= 0.0 && d2 <= 0.0) return a; // vertex a

        const RowVector3 bp = p - b;
        const Scalar d3 = ab.dot(bp), d4 = ac.dot(bp);
        if (d3 >= 0.0 && d4 <= d3) return b; // vertex b

        const Scalar vc = d1 * d4 - d3 * d2;
        if (a != b && vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { // edge ab
            const Scalar v = d1 / (d1 - d3);
            return a + v * ab;
        }
        const RowVector3 cp = p - c;
        const Scalar d5 = ab.dot(cp), d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6) return c; // vertex c

        const Scalar vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { // edge ac
            const Scalar w = d2 / (d2 - d6);
            return a + w * ac;
        }
        const Scalar va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) { // edge bc
            const Scalar w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + w * (c - b);
        }
        const Scalar denom = 1.0 / (va + vb + vc); // face
        const Scalar v = vb * denom, w = vc * denom;
        return a + ab * v + ac * w;
    }
};
