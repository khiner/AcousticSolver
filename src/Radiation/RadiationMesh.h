#pragma once

// Flat-triangle boundary mesh for the radiation solver: element-constant (P0) data lives
// at centroids, normals point out of the body into the fluid. The solver requires a closed
// manifold whose elements are no larger than a grid cell (the paper's constraint, which
// keeps an element's near set well defined and its interpolation stencil local), so scene
// meshes come through Remesh. Icosphere generation serves the analytic monopole tests.

#include <Eigen/Core>

#include <string>
#include <vector>

struct RadiationMesh {
    std::vector<Eigen::Vector3d> Vertices;
    std::vector<Eigen::Vector3i> Tris;

    // Per-element derived data, filled by Finalize().
    std::vector<Eigen::Vector3d> Centroids, Normals;
    std::vector<double> Areas;
    double MaxEdge{0.};

    int NumElems() const { return int(Tris.size()); }
    // Derives centroids, areas, and outward normals. Faces must be consistently wound;
    // the global orientation comes from the enclosed signed volume, so the normals point
    // out of the body whichever way the source file wound it.
    void Finalize();

    // True when the elements share at least one vertex (the paper's 3-point quadrature case).
    bool Adjacent(int a, int b) const;

    // Every edge shared by exactly two triangles, in opposite directions. The solver's
    // exterior representation assumes it.
    bool IsClosed() const;

    // Retargets the mesh to the grid by incremental isotropic remeshing towards edge
    // length `target`, with `max_edge` as a hard ceiling. Preserves closedness and
    // orientation and leaves the mesh finalized.
    void Remesh(double target, double max_edge);

    // Fraction of the ceiling to aim for. Measured on the scene meshes: aiming higher
    // leaves edges above the ceiling for the final split pass to halve, which ends up
    // *more* elements, and aiming lower simply resolves finer than the ceiling asks.
    static constexpr double RemeshTarget{0.6};

    static RadiationMesh Icosphere(const Eigen::Vector3d &center, double radius, int subdiv);
    static RadiationMesh FromObj(const std::string &path, const Eigen::Vector3d &offset);
};
