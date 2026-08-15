#pragma once

// Ported from WaveBlender (c) 2024 Kangrui Xue (WaveBlender.h) — Metal port.
//
// NOTE: Ordering of objects matters. Point sources MUST be placed at end; later objects
// will override previous rasterizations.

#include "MetalSolver.h"

constexpr int CavityInterior{255};

// Extends FDTDSolver with the gritty implementation details (e.g., fresh cell extrapolation)
// needed to make the WaveBlender system work.
class WaveBlender : public FDTDSolver {
public:
    WaveBlender(const SimParams &params) : FDTDSolver(params) {
        Cell1.resize(GridSize);
        Cell2.resize(GridSize);
    }

    // Runs a complete simulation batch: from rasterization to FDTD timestepping.
    bool RunBatch();

    void AddObject(Eigen::Vector3<REAL> offset, std::shared_ptr<Object> object) {
        Offsets.emplace_back(offset);
        Objects.emplace_back(std::move(object));
    }

private:
    std::vector<Eigen::Vector3<REAL>> Offsets;
    std::vector<std::shared_ptr<Object>> Objects;

    std::vector<int> Cell1, Cell2; // rasterized cell states at batch endpoints t1 and t2

    // ----- Rasterization -----
    void Rasterize(const std::set<int> &unchanged_oids, bool log_raster = false);
    void SetupShaders();

    // ----- Per-batch overhead -----
    void DetectCavities();
    void FreshCellPressure();
    void FreshCellVelocity();
    void ShaderReInit();
};
