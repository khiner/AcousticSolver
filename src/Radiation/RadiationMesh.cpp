#include "RadiationMesh.h"

#include "AabbTree.h"
#include "Mesh.h"

#include <Eigen/Geometry>

#include <cmath>
#include <map>
#include <numbers>
#include <queue>

using Eigen::Vector3d, Eigen::Vector3i;

void RadiationMesh::Finalize() {
    const auto m = Tris.size();
    Centroids.resize(m);
    Normals.resize(m);
    Areas.resize(m);
    MaxEdge = 0.;
    // Six times the enclosed signed volume, from the divergence theorem over the faces.
    double signed_volume = 0.;
    for (size_t e = 0; e < m; ++e) {
        const Vector3d a = Vertices[Tris[e][0]], b = Vertices[Tris[e][1]], c = Vertices[Tris[e][2]];
        Centroids[e] = (a + b + c) / 3.;
        const Vector3d cr = (b - a).cross(c - a);
        const double two_area = cr.norm();
        Areas[e] = 0.5 * two_area;
        Normals[e] = two_area > 0. ? Vector3d{cr / two_area} : Vector3d::UnitZ();
        signed_volume += a.dot(cr);
        MaxEdge = std::max({MaxEdge, (b - a).norm(), (c - b).norm(), (a - c).norm()});
    }
    if (signed_volume < 0.) {
        for (auto &n : Normals) n = -n;
    }
}

bool RadiationMesh::IsClosed() const {
    std::map<std::pair<int, int>, int> directed;
    for (const auto &tri : Tris) {
        for (int i = 0; i < 3; ++i) {
            const int a = tri[i], b = tri[(i + 1) % 3];
            if (a == b) return false;
            if (++directed[{a, b}] > 1) return false; // an edge used twice the same way
        }
    }
    for (const auto &[edge, count] : directed) {
        if (!directed.contains({edge.second, edge.first})) return false;
    }
    return true;
}

bool RadiationMesh::Adjacent(int a, int b) const {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (Tris[a][i] == Tris[b][j]) return true;
    return false;
}

RadiationMesh RadiationMesh::Icosphere(const Vector3d &center, double radius, int subdiv) {
    RadiationMesh mesh;
    const double t = std::numbers::phi;
    const Vector3d base[12]{
        {-1, t, 0},
        {1, t, 0},
        {-1, -t, 0},
        {1, -t, 0},
        {0, -1, t},
        {0, 1, t},
        {0, -1, -t},
        {0, 1, -t},
        {t, 0, -1},
        {t, 0, 1},
        {-t, 0, -1},
        {-t, 0, 1},
    };
    for (const auto &v : base) mesh.Vertices.push_back(v.normalized());
    static constexpr int Faces[20][3]{
        {0, 11, 5},
        {0, 5, 1},
        {0, 1, 7},
        {0, 7, 10},
        {0, 10, 11},
        {1, 5, 9},
        {5, 11, 4},
        {11, 10, 2},
        {10, 7, 6},
        {7, 1, 8},
        {3, 9, 4},
        {3, 4, 2},
        {3, 2, 6},
        {3, 6, 8},
        {3, 8, 9},
        {4, 9, 5},
        {2, 4, 11},
        {6, 2, 10},
        {8, 6, 7},
        {9, 8, 1},
    };
    for (const auto &f : Faces) mesh.Tris.emplace_back(f[0], f[1], f[2]);

    for (int level = 0; level < subdiv; ++level) {
        std::map<std::pair<int, int>, int> midpoints;
        const auto midpoint = [&](int a, int b) {
            const auto key = std::minmax(a, b);
            if (const auto it = midpoints.find(key); it != midpoints.end()) return it->second;
            mesh.Vertices.push_back((mesh.Vertices[a] + mesh.Vertices[b]).normalized());
            const int idx = int(mesh.Vertices.size()) - 1;
            midpoints.emplace(key, idx);
            return idx;
        };
        std::vector<Vector3i> next;
        next.reserve(mesh.Tris.size() * 4);
        for (const auto &tri : mesh.Tris) {
            const int ab = midpoint(tri[0], tri[1]), bc = midpoint(tri[1], tri[2]), ca = midpoint(tri[2], tri[0]);
            next.emplace_back(tri[0], ab, ca);
            next.emplace_back(tri[1], bc, ab);
            next.emplace_back(tri[2], ca, bc);
            next.emplace_back(ab, bc, ca);
        }
        mesh.Tris = std::move(next);
    }
    for (auto &v : mesh.Vertices) v = center + radius * v;
    mesh.Finalize();
    return mesh;
}

RadiationMesh RadiationMesh::FromObj(const std::string &path, const Vector3d &offset) {
    Eigen::MatrixXd v;
    Eigen::MatrixXi f;
    if (!ReadObj(path, v, f)) throw std::runtime_error("Failed to read mesh: " + path);
    RadiationMesh mesh;
    mesh.Vertices.reserve(v.rows());
    for (long r = 0; r < v.rows(); ++r) mesh.Vertices.emplace_back(v(r, 0) + offset[0], v(r, 1) + offset[1], v(r, 2) + offset[2]);
    mesh.Tris.reserve(f.rows());
    for (long r = 0; r < f.rows(); ++r) mesh.Tris.emplace_back(f(r, 0), f(r, 1), f(r, 2));
    mesh.Finalize();
    return mesh;
}

namespace {
// Unit normal of a triangle, zero when it is degenerate.
Vector3d TriNormal(const Vector3d &a, const Vector3d &b, const Vector3d &c) {
    const Vector3d cr = (b - a).cross(c - a);
    const double n = cr.norm();
    return n > 0. ? Vector3d{cr / n} : Vector3d::Zero();
}

// The fundamental error quadric of a triangle's plane, as in [Garland & Heckbert 1997],
// scaled by area so large faces dominate the placement.
Eigen::Matrix4d PlaneQuadric(const Vector3d &a, const Vector3d &b, const Vector3d &c) {
    const Vector3d cr = (b - a).cross(c - a);
    const double two_area = cr.norm();
    if (two_area <= 0.) return Eigen::Matrix4d::Zero();
    const Vector3d n = cr / two_area;
    Eigen::Vector4d p;
    p << n, -n.dot(a);
    return 0.5 * two_area * (p * p.transpose());
}

// Working half-edge-free mesh: faces plus per-vertex incident-face lists, both with
// tombstones. Every operation preserves closedness and winding.
struct EditMesh {
    std::vector<Vector3d> V;
    std::vector<Eigen::Vector3i> F;
    std::vector<uint8_t> FaceAlive, VertAlive;
    std::vector<std::vector<int>> VertFaces;
    std::vector<Eigen::Matrix4d> Q;

    void Build(const RadiationMesh &mesh) {
        V = mesh.Vertices;
        F = mesh.Tris;
        FaceAlive.assign(F.size(), 1);
        VertAlive.assign(V.size(), 0);
        VertFaces.assign(V.size(), {});
        Q.assign(V.size(), Eigen::Matrix4d::Zero());
        for (size_t f = 0; f < F.size(); ++f) {
            const auto quadric = PlaneQuadric(V[F[f][0]], V[F[f][1]], V[F[f][2]]);
            for (int i = 0; i < 3; ++i) {
                const int v = F[f][i];
                VertFaces[v].push_back(int(f));
                VertAlive[v] = 1;
                Q[v] += quadric;
            }
        }
    }

    // Drops dead and stale entries from a vertex's face list, in place.
    void Compact(int v) {
        auto &list = VertFaces[v];
        size_t out = 0;
        for (const int f : list) {
            if (FaceAlive[f] && (F[f][0] == v || F[f][1] == v || F[f][2] == v)) list[out++] = f;
        }
        list.resize(out);
    }

    double Area(int f) const { return 0.5 * (V[F[f][1]] - V[F[f][0]]).cross(V[F[f][2]] - V[F[f][0]]).norm(); }
    Vector3d Normal(int f) const { return TriNormal(V[F[f][0]], V[F[f][1]], V[F[f][2]]); }

    // The two faces sharing an edge, or false when the edge is not manifold.
    bool SharedFaces(int u, int v, int &f0, int &f1) {
        Compact(u); // stale entries would be miscounted as extra incident faces
        f0 = f1 = -1;
        for (const int f : VertFaces[u]) {
            if (!FaceAlive[f]) continue;
            if (F[f][0] != v && F[f][1] != v && F[f][2] != v) continue;
            if (f0 < 0) f0 = f;
            else if (f1 < 0) f1 = f;
            else return false;
        }
        return f1 >= 0;
    }

    // Collapse u into v, placing the survivor at `at`. Rejects collapses that would tear
    // the surface (link condition) or fold a face over.
    bool Collapse(int u, int v, const Vector3d &at) {
        int f0 = -1, f1 = -1;
        if (!SharedFaces(u, v, f0, f1)) return false;
        Compact(v);

        // Link condition: the only vertices adjacent to both endpoints must be the two
        // opposite corners of the shared faces.
        std::vector<int> shared_opposite;
        for (const int f : {f0, f1}) {
            for (int i = 0; i < 3; ++i) {
                if (F[f][i] != u && F[f][i] != v) shared_opposite.push_back(F[f][i]);
            }
        }
        for (const int fu : VertFaces[u]) {
            for (int i = 0; i < 3; ++i) {
                const int w = F[fu][i];
                if (w == u || w == v) continue;
                bool adj_v = false;
                for (const int fv : VertFaces[v]) {
                    if (F[fv][0] == w || F[fv][1] == w || F[fv][2] == w) {
                        adj_v = true;
                        break;
                    }
                }
                if (adj_v && std::ranges::find(shared_opposite, w) == shared_opposite.end()) return false;
            }
        }

        // Orientation and degeneracy check on every surviving incident face.
        const Vector3d old_u = V[u], old_v = V[v];
        V[u] = V[v] = at;
        bool ok = true;
        for (const int side : {u, v}) {
            for (const int face : VertFaces[side]) {
                if (face == f0 || face == f1) continue;
                if (Area(face) < 1e-18 || OldNormal(face, u, v, old_u, old_v).dot(Normal(face)) < 0.2) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
        if (!ok) {
            V[u] = old_u;
            V[v] = old_v;
            return false;
        }

        FaceAlive[f0] = FaceAlive[f1] = 0;
        for (const int f : VertFaces[u]) {
            if (!FaceAlive[f]) continue;
            for (int i = 0; i < 3; ++i) {
                if (F[f][i] == u) F[f][i] = v;
            }
            VertFaces[v].push_back(f);
        }
        VertFaces[u].clear();
        VertAlive[u] = 0;
        Q[v] += Q[u];
        Compact(v);
        return true;
    }

    // Normal of `face` before the endpoints moved, for the fold test.
    Vector3d OldNormal(int face, int u, int v, const Vector3d &old_u, const Vector3d &old_v) const {
        Vector3d p[3];
        for (int i = 0; i < 3; ++i) {
            const int w = F[face][i];
            p[i] = w == u ? old_u : (w == v ? old_v : V[w]);
        }
        return TriNormal(p[0], p[1], p[2]);
    }

    // Splits edge (u, v) at its midpoint, re-triangulating both incident faces.
    bool Split(int u, int v) {
        int f0 = -1, f1 = -1;
        if (!SharedFaces(u, v, f0, f1)) return false;
        const int w = int(V.size());
        V.emplace_back(0.5 * (V[u] + V[v]));
        VertAlive.push_back(1);
        VertFaces.emplace_back();
        Q.emplace_back(0.5 * (Q[u] + Q[v]));
        for (const int f : {f0, f1}) {
            // Rotate the face so its (u, v) edge (in winding order) starts at index 0.
            Eigen::Vector3i tri = F[f];
            int rot = 0;
            for (int i = 0; i < 3; ++i) {
                if ((tri[i] == u && tri[(i + 1) % 3] == v) || (tri[i] == v && tri[(i + 1) % 3] == u)) rot = i;
            }
            const int a = tri[rot], b = tri[(rot + 1) % 3], c = tri[(rot + 2) % 3];
            F[f] = {a, w, c};
            const int nf = int(F.size());
            F.emplace_back(w, b, c);
            FaceAlive.push_back(1);
            VertFaces[w].push_back(f);
            VertFaces[w].push_back(nf);
            VertFaces[b].push_back(nf);
            VertFaces[c].push_back(nf);
        }
        return true;
    }

    // Valence of a vertex: on a closed manifold it equals its incident-face count.
    int Valence(int v) {
        Compact(v);
        return int(VertFaces[v].size());
    }

    bool Connected(int a, int b) {
        Compact(a);
        for (const int f : VertFaces[a]) {
            if (F[f][0] == b || F[f][1] == b || F[f][2] == b) return true;
        }
        return false;
    }

    // Rotates the shared edge of the two faces on (u, v) onto their opposite corners.
    // Rejects flips that would duplicate an edge or fold either new face over.
    bool Flip(int u, int v) {
        int f0 = -1, f1 = -1;
        if (!SharedFaces(u, v, f0, f1)) return false;
        const auto rotated = [&](int f, int first, int second) { // face starts at `first` with `second` next
            for (int i = 0; i < 3; ++i) {
                if (F[f][i] == first && F[f][(i + 1) % 3] == second) return Eigen::Vector3i{F[f][i], F[f][(i + 1) % 3], F[f][(i + 2) % 3]};
            }
            return Eigen::Vector3i{-1, -1, -1};
        };
        Eigen::Vector3i t0 = rotated(f0, u, v);
        if (t0[0] < 0) {
            std::swap(f0, f1);
            t0 = rotated(f0, u, v);
        }
        const Eigen::Vector3i t1 = rotated(f1, v, u);
        if (t0[0] < 0 || t1[0] < 0) return false;
        const int a = t0[2], b = t1[2];
        if (a == b || Connected(a, b)) return false;

        const Vector3d n0 = Normal(f0), n1 = Normal(f1);
        const Eigen::Vector3i new0{u, b, a}, new1{v, a, b};
        const Eigen::Vector3i old0 = F[f0], old1 = F[f1];
        F[f0] = new0;
        F[f1] = new1;
        if (Area(f0) < 1e-18 || Area(f1) < 1e-18 || Normal(f0).dot(n0) < 0.5 || Normal(f1).dot(n1) < 0.5) {
            F[f0] = old0;
            F[f1] = old1;
            return false;
        }
        VertFaces[b].push_back(f0);
        VertFaces[a].push_back(f1);
        return true;
    }

    // One-ring neighbours of a vertex.
    void Neighbors(int v, std::vector<int> &out) {
        Compact(v);
        out.clear();
        for (const int f : VertFaces[v]) {
            for (int i = 0; i < 3; ++i) {
                if (F[f][i] != v) out.push_back(F[f][i]);
            }
        }
        std::ranges::sort(out);
        out.erase(std::ranges::unique(out).begin(), out.end());
    }

    // Area-weighted vertex normal, for the tangential projection of the smoothing step.
    Vector3d VertexNormal(int v) {
        Compact(v);
        Vector3d n = Vector3d::Zero();
        for (const int f : VertFaces[v]) n += Area(f) * Normal(f);
        const double len = n.norm();
        return len > 0. ? Vector3d{n / len} : Vector3d::Zero();
    }

    void Emit(RadiationMesh &mesh) const {
        std::vector<int> remap(V.size(), -1);
        mesh.Vertices.clear();
        mesh.Tris.clear();
        for (size_t f = 0; f < F.size(); ++f) {
            if (!FaceAlive[f]) continue;
            Eigen::Vector3i tri;
            for (int i = 0; i < 3; ++i) {
                const int v = F[f][i];
                if (remap[v] < 0) {
                    remap[v] = int(mesh.Vertices.size());
                    mesh.Vertices.push_back(V[v]);
                }
                tri[i] = remap[v];
            }
            mesh.Tris.push_back(tri);
        }
    }
};

// Quadric-optimal survivor position, falling back to the midpoint when the system is
// ill-conditioned (flat or symmetric neighborhoods).
Vector3d OptimalPoint(const Eigen::Matrix4d &q, const Vector3d &a, const Vector3d &b) {
    Eigen::Matrix3d m = q.topLeftCorner<3, 3>();
    const Eigen::Vector3d rhs = -q.topRightCorner<3, 1>();
    const Eigen::FullPivLU<Eigen::Matrix3d> lu{m};
    if (lu.rank() == 3) {
        const Vector3d x = lu.solve(rhs);
        const Vector3d mid = 0.5 * (a + b);
        if ((x - mid).norm() < 4. * (a - b).norm()) return x;
    }
    return 0.5 * (a + b);
}
} // namespace

void RadiationMesh::Remesh(double target, double max_edge) {
    // Incremental isotropic remeshing [Botsch & Kobbelt 2004]: split, collapse, flip for
    // valence, relax tangentially, and project back onto the input surface. Collapse alone
    // stalls well above the element count the ceiling allows, and the flips are what unblock
    // it — which matters because the tables scale on that count.
    AabbTree<double> reference;
    {
        Eigen::MatrixXd v(Vertices.size(), 3);
        Eigen::MatrixXi f(Tris.size(), 3);
        for (size_t i = 0; i < Vertices.size(); ++i) v.row(i) = Vertices[i].transpose();
        for (size_t i = 0; i < Tris.size(); ++i) f.row(i) = Tris[i].transpose();
        reference.Init(v, f);
    }
    EditMesh em;
    em.Build(*this);

    const double split_at = std::min(4. / 3. * target, max_edge), collapse_at = 0.8 * target;
    std::vector<uint32_t> stamp;
    struct Cand {
        double Key;
        int U, V;
        uint32_t StampU, StampV;
        bool operator<(const Cand &o) const { return Key > o.Key; } // min-heap
    };

    // Every edge of every live face, in face order. Each interior edge arrives twice, once
    // from each side, so callers that want it once filter on u < v.
    const auto for_each_edge = [&](auto &&fn) {
        for (size_t f = 0; f < em.F.size(); ++f) {
            if (!em.FaceAlive[f]) continue;
            for (int i = 0; i < 3; ++i) fn(em.F[f][i], em.F[f][(i + 1) % 3]);
        }
    };

    // A sweep can take every long edge it finds: a split only re-triangulates the two faces
    // on its own edge and never lengthens another.
    const auto split_pass = [&](double ceiling) {
        for (int pass = 0; pass < 24; ++pass) {
            std::vector<std::pair<int, int>> longs;
            for_each_edge([&](int u, int v) {
                if (u < v && (em.V[u] - em.V[v]).norm() > ceiling) longs.emplace_back(u, v);
            });
            if (longs.empty()) break;
            for (const auto &[u, v] : longs) em.Split(u, v);
        }
    };

    // Shortest edges first, so the survivors spread out evenly. Stamps invalidate queue
    // entries whose endpoints have since moved or been removed.
    const auto collapse_pass = [&] {
        stamp.assign(em.V.size(), 0);
        std::priority_queue<Cand> queue;
        const auto push_edge = [&](int u, int v) {
            const double len = (em.V[u] - em.V[v]).norm();
            if (len >= collapse_at) return;
            queue.push({len, u, v, stamp[u], stamp[v]});
        };
        for_each_edge(push_edge);
        while (!queue.empty()) {
            const Cand cand = queue.top();
            queue.pop();
            const int u = cand.U, v = cand.V;
            if (!em.VertAlive[u] || !em.VertAlive[v]) continue;
            if (stamp[u] != cand.StampU || stamp[v] != cand.StampV) continue;
            if ((em.V[u] - em.V[v]).norm() >= collapse_at) continue;
            const Vector3d mid = 0.5 * (em.V[u] + em.V[v]), best = OptimalPoint(em.Q[u] + em.Q[v], em.V[u], em.V[v]);
            // The quadric's own minimum can fold a neighbouring face on thin geometry; the
            // midpoint rarely does, and taking it keeps the element count near the target.
            if (!em.Collapse(u, v, best) && (best == mid || !em.Collapse(u, v, mid))) continue;
            // Only the two endpoints changed, so only entries naming them go stale. Stamping
            // the whole one-ring would drop its outward edges from the queue for good, which
            // is what stalls the collapse well above the target length.
            ++stamp[u];
            ++stamp[v];
            em.Compact(v);
            for (const int f : em.VertFaces[v]) {
                for (int i = 0; i < 3; ++i) push_edge(em.F[f][i], em.F[f][(i + 1) % 3]);
            }
        }
    };

    const auto flip_pass = [&] {
        std::vector<std::pair<int, int>> edges;
        for_each_edge([&](int u, int v) {
            if (u < v) edges.emplace_back(u, v);
        });
        for (const auto &[u, v] : edges) {
            if (!em.VertAlive[u] || !em.VertAlive[v]) continue;
            int f0 = -1, f1 = -1;
            if (!em.SharedFaces(u, v, f0, f1)) continue;
            int a = -1, b = -1;
            for (int i = 0; i < 3; ++i) {
                if (em.F[f0][i] != u && em.F[f0][i] != v) a = em.F[f0][i];
                if (em.F[f1][i] != u && em.F[f1][i] != v) b = em.F[f1][i];
            }
            if (a < 0 || b < 0) continue;
            const int vu = em.Valence(u), vv = em.Valence(v), va = em.Valence(a), vb = em.Valence(b);
            const int before = std::abs(vu - 6) + std::abs(vv - 6) + std::abs(va - 6) + std::abs(vb - 6);
            const int after = std::abs(vu - 7) + std::abs(vv - 7) + std::abs(va - 5) + std::abs(vb - 5);
            if (after < before) em.Flip(u, v);
        }
    };

    // Tangential relaxation, then back onto the input surface: moving along the tangent
    // plane evens out the element sizes without drifting off the geometry.
    const auto smooth_pass = [&] {
        std::vector<Vector3d> moved(em.V.size());
        std::vector<int> ring;
        for (size_t v = 0; v < em.V.size(); ++v) {
            moved[v] = em.V[v];
            if (!em.VertAlive[v]) continue;
            em.Neighbors(int(v), ring);
            if (ring.empty()) continue;
            Vector3d centroid = Vector3d::Zero();
            for (const int w : ring) centroid += em.V[w];
            centroid /= double(ring.size());
            const Vector3d n = em.VertexNormal(int(v));
            Vector3d delta = centroid - em.V[v];
            delta -= n.dot(delta) * n;
            moved[v] = em.V[v] + 0.5 * delta;
        }
        for (size_t v = 0; v < em.V.size(); ++v) {
            if (!em.VertAlive[v]) continue;
            int tri = 0;
            Eigen::RowVector3d closest;
            reference.ClosestPoint(moved[v].transpose(), tri, closest);
            em.V[v] = closest.transpose();
        }
    };

    for (int iter = 0; iter < 5; ++iter) {
        split_pass(split_at);
        collapse_pass();
        flip_pass();
        smooth_pass();
    }
    split_pass(max_edge); // the ceiling is a hard constraint, whatever relaxation did

    em.Emit(*this);
    Finalize();
}
