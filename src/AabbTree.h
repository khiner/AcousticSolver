#pragma once

// Axis-aligned bounding box hierarchy over a triangle mesh, for closest-point queries.
// The split rule (median barycenter rank along the box's longest axis) and the child-visit
// order mirror the reference implementation, so queries resolve to the triangle it picks.
//
// Layout is traversal-friendly: an interior node stores both children's boxes (so a visit
// decision touches one node), and leaf triangle vertices are copied inline at Init so a
// triangle test reads one contiguous record instead of gathering rows of the mesh.

#include <Eigen/Core>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

template<typename Scalar> struct AabbTree {
    using Vertices = Eigen::MatrixX<Scalar>;
    using RowVector3 = Eigen::Matrix<Scalar, 1, 3>;

    void Init(const Vertices &v, const Eigen::MatrixXi &f) {
        Nodes.clear();
        Leaves.clear();
        Root = -1;
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
        Nodes.reserve(f.rows());
        Leaves.reserve(f.rows());
        Box root_box;
        Root = Build(v, f, ranks, elements, root_box);
    }

    // Returns the first triangle crossed by [a, b], or -1.
    int Crossing(const RowVector3 &a, const RowVector3 &b, Scalar &t_out) const {
        int i = -1;
        t_out = std::numeric_limits<Scalar>::infinity();
        if (Leaves.empty()) return -1;
        VisitSegment(Root, a, b - a, t_out, i);
        return i;
    }

    // Index of the triangle closest to `p`, and the closest point on it.
    void ClosestPoint(const RowVector3 &p, int &i_out, RowVector3 &c_out) const {
        i_out = -1;
        c_out.setZero();
        if (Leaves.empty()) return;

        Scalar sqr_d = std::numeric_limits<Scalar>::infinity();
        Visit(Root, p, sqr_d, i_out, c_out);
    }

private:
    struct Box {
        Scalar Min[3], Max[3];
    };
    // Child references: a non-negative ref indexes Nodes, a negative ref is ~index into Leaves.
    struct Node {
        Box LBox, RBox;
        int LRef, RRef;
    };
    struct Leaf {
        RowVector3 A, B, C; // triangle vertices, copied at Init
        int Triangle; // original triangle index
    };

    int Root{-1};
    std::vector<Node> Nodes;
    std::vector<Leaf> Leaves;

    // Returns the new child's ref and writes its bounding box (over `elements`) to `box`.
    int Build(const Vertices &v, const Eigen::MatrixXi &f, const Eigen::MatrixXi &ranks, const std::vector<int> &elements, Box &box) {
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
        std::copy_n(lo, 3, box.Min);
        std::copy_n(hi, 3, box.Max);

        if (elements.size() == 1) {
            const int e = elements[0];
            Leaves.push_back({v.row(f(e, 0)), v.row(f(e, 1)), v.row(f(e, 2)), e});
            return ~int(Leaves.size() - 1);
        }

        int axis = 0;
        for (int d = 1; d < 3; ++d) {
            if (hi[d] - lo[d] > hi[axis] - lo[axis]) axis = d;
        }
        std::vector<int> nth(elements.size());
        for (size_t i = 0; i < elements.size(); ++i) nth[i] = ranks(elements[i], axis);
        const size_t mid = (nth.size() - 1) / 2;
        std::nth_element(nth.begin(), nth.begin() + mid, nth.end());
        const int median = nth[mid];

        // Ranks are distinct, so this always splits into two non-empty halves.
        std::vector<int> left, right;
        left.reserve((elements.size() + 1) / 2);
        right.reserve(elements.size() / 2);
        for (const int e : elements) (ranks(e, axis) <= median ? left : right).push_back(e);

        const int node = int(Nodes.size());
        Nodes.emplace_back();
        Box l_box, r_box;
        const int l = Build(v, f, ranks, left, l_box);
        const int r = Build(v, f, ranks, right, r_box);
        Nodes[node] = {l_box, r_box, l, r};
        return node;
    }

    static bool SegmentBox(const Box &box, const RowVector3 &o, const RowVector3 &d, Scalar limit, Scalar &entry) {
        Scalar lo = 0, hi = std::min(Scalar(1), limit);
        for (int axis = 0; axis < 3; ++axis) {
            if (d[axis] == Scalar(0)) {
                if (o[axis] < box.Min[axis] || o[axis] > box.Max[axis]) return false;
                continue;
            }
            Scalar a = (box.Min[axis] - o[axis]) / d[axis];
            Scalar b = (box.Max[axis] - o[axis]) / d[axis];
            if (a > b) std::swap(a, b);
            lo = std::max(lo, a);
            hi = std::min(hi, b);
            if (lo > hi) return false;
        }
        entry = lo;
        return true;
    }

    // Avoid <Eigen/Geometry> in this Core-only header.
    static RowVector3 Cross(const RowVector3 &u, const RowVector3 &v) {
        return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    }

    static bool SegmentTriangle(const RowVector3 &o, const RowVector3 &d, const RowVector3 &a, const RowVector3 &b, const RowVector3 &c, Scalar &t) {
        const RowVector3 e1 = b - a, e2 = c - a, p = Cross(d, e2);
        const Scalar det = e1.dot(p);
        if (std::abs(det) <= std::numeric_limits<Scalar>::min()) return false;
        const Scalar inv = Scalar(1) / det;
        const RowVector3 s = o - a;
        const Scalar u = s.dot(p) * inv;
        if (u < Scalar(0) || u > Scalar(1)) return false;
        const RowVector3 q = Cross(s, e1);
        const Scalar v = d.dot(q) * inv;
        if (v < Scalar(0) || u + v > Scalar(1)) return false;
        t = e2.dot(q) * inv;
        return t >= Scalar(0) && t <= Scalar(1);
    }

    void VisitSegment(int ref, const RowVector3 &o, const RowVector3 &d, Scalar &t, int &i) const {
        if (ref < 0) {
            const Leaf &leaf = Leaves[~ref];
            Scalar hit;
            if (SegmentTriangle(o, d, leaf.A, leaf.B, leaf.C, hit) && hit < t) {
                t = hit;
                i = leaf.Triangle;
            }
            return;
        }
        const Node &node = Nodes[ref];
        Scalar left, right;
        const bool visit_left = SegmentBox(node.LBox, o, d, t, left);
        const bool visit_right = SegmentBox(node.RBox, o, d, t, right);
        if (visit_left && visit_right) {
            if (left <= right) {
                VisitSegment(node.LRef, o, d, t, i);
                if (right < t) VisitSegment(node.RRef, o, d, t, i);
            } else {
                VisitSegment(node.RRef, o, d, t, i);
                if (left < t) VisitSegment(node.LRef, o, d, t, i);
            }
        } else if (visit_left) VisitSegment(node.LRef, o, d, t, i);
        else if (visit_right) VisitSegment(node.RRef, o, d, t, i);
    }

    void Visit(int ref, const RowVector3 &p, Scalar &sqr_d, int &i, RowVector3 &c) const {
        if (ref < 0) {
            const Leaf &leaf = Leaves[~ref];
            const RowVector3 candidate = ClosestPointOnTriangle(p, leaf.A, leaf.B, leaf.C);
            const Scalar d = (p - candidate).squaredNorm();
            if (d < sqr_d) {
                sqr_d = d;
                i = leaf.Triangle;
                c = candidate;
            }
            return;
        }
        // Descend into any child whose box contains the query point, then into the
        // remaining ones nearest-first, skipping those that cannot beat the current best.
        const Node &n = Nodes[ref];
        bool looked_left = false, looked_right = false;
        if (Contains(n.LBox, p)) {
            Visit(n.LRef, p, sqr_d, i, c);
            looked_left = true;
        }
        if (Contains(n.RBox, p)) {
            Visit(n.RRef, p, sqr_d, i, c);
            looked_right = true;
        }
        if (looked_left && looked_right) return;

        const Scalar left_d = ExteriorSquaredDistance(n.LBox, p);
        const Scalar right_d = ExteriorSquaredDistance(n.RBox, p);
        if (left_d < right_d) {
            if (!looked_left && left_d < sqr_d) Visit(n.LRef, p, sqr_d, i, c);
            if (!looked_right && right_d < sqr_d) Visit(n.RRef, p, sqr_d, i, c);
        } else {
            if (!looked_right && right_d < sqr_d) Visit(n.RRef, p, sqr_d, i, c);
            if (!looked_left && left_d < sqr_d) Visit(n.LRef, p, sqr_d, i, c);
        }
    }

    static bool Contains(const Box &b, const RowVector3 &p) {
        for (int d = 0; d < 3; ++d) {
            if (p[d] < b.Min[d] || p[d] > b.Max[d]) return false;
        }
        return true;
    }

    static Scalar ExteriorSquaredDistance(const Box &b, const RowVector3 &p) {
        Scalar sqr_d{0};
        for (int d = 0; d < 3; ++d) {
            if (b.Min[d] > p[d]) {
                const Scalar aux = b.Min[d] - p[d];
                sqr_d += aux * aux;
            } else if (p[d] > b.Max[d]) {
                const Scalar aux = p[d] - b.Max[d];
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
