#pragma once

// (c) 2024 Kangrui Xue, (c) 2023 Jui-Hsien Wang. Adapted from ModalSound / openpbso (MIT) — see NOTICE.md.
// Modal vibration of a rigid body, driven by recorded contact impulses.
//
// Based on code by Jui-Hsien Wang (https://github.com/jhwang7628/openpbso)
//
// TODO: in general, this library is still pretty barebones (and in need of cleanup)

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

namespace ModalSound {

struct ImpactRecord {
    Eigen::Vector3d ImpactVector;
    Eigen::Vector3d ImpactPosition; // In object space
    double Timestamp{0};
    double SupportLength{0.01}; // tau
    double ContactSpeed{0};
    double Gamma{1.}; // TODO: better default, gamma = pi * norm(J) / (2 tau)
    int AppliedVertex{0};
};

// The half-sine contact profile at `time`, zero outside the impact's support. Scales both
// the modal force an impact injects and the rigid-body acceleration noise it produces
// (see AccelPulse), on both solvers' modal paths.
inline double ContactProfile(const ImpactRecord &r, double time) {
    if (time < r.Timestamp || time >= r.Timestamp + r.SupportLength) return 0.;
    return std::sin(std::numbers::pi * (time - r.Timestamp) / r.SupportLength);
}

struct ImpulseSeries {
    // The vertex index must be into the surface triangle mesh, not the volumetric tet mesh.
    void Add(const ImpactRecord &record) {
        LastTime = record.Timestamp;
        FirstTime = std::min(FirstTime, record.Timestamp);
        Records.push_back(record);
    }

    // The impacts whose support contains `time`, with their impulse vectors scaled by the
    // half-sine contact profile. When the records are sorted by timestamp (they are, for
    // every impulse file we load), only the window that can overlap `time` is scanned.
    void GetForces(double time, std::vector<ImpactRecord> &out) const {
        out.clear();
        auto begin = Records.begin(), end = Records.end();
        if (SortedByTime) {
            begin = std::lower_bound(begin, end, time, [max = MaxSupportLength](const ImpactRecord &r, double t) { return r.Timestamp + max <= t; });
            end = std::upper_bound(begin, end, time, [](double t, const ImpactRecord &r) { return t < r.Timestamp; });
        }
        for (auto it = begin; it != end; ++it) {
            if (time >= it->Timestamp && time < it->Timestamp + it->SupportLength) out.push_back(*it);
        }
        for (auto &r : out) {
            const double j = r.ImpactVector.norm();
            r.ImpactVector = (r.ImpactVector / j) * r.Gamma;

            r.ImpactVector *= ContactProfile(r, time);
        }
    }

    // Drops impulses whose contact speed is too low to be audible.
    void Filter() {
        static constexpr double VelocityThreshold{0.05};
        std::erase_if(Records, [](const ImpactRecord &r) { return std::abs(r.ContactSpeed) < VelocityThreshold; });

        if (Records.empty()) {
            FirstTime = LastTime = 0.;
        } else {
            FirstTime = Records.front().Timestamp;
            LastTime = Records.size() == 1 ? FirstTime + 1e-8 : Records.back().Timestamp;
        }

        MaxSupportLength = 0.;
        for (const auto &r : Records) MaxSupportLength = std::max(MaxSupportLength, r.SupportLength);
        SortedByTime = std::is_sorted(Records.begin(), Records.end(), [](const ImpactRecord &a, const ImpactRecord &b) { return a.Timestamp < b.Timestamp; });
    }

    std::vector<ImpactRecord> Records;
    double FirstTime{0}, LastTime{0};
    double MaxSupportLength{0};
    bool SortedByTime{false};
};

struct Solver {
    Solver(const std::string &data_prefix, const std::string &mesh_file, const std::string &material_name, double dt);

    // Timesteps the modal oscillators.
    void Step(double time);

    Eigen::MatrixXd EigenVectorsNormal; // Eigenvectors projected onto the vertex normal, N_S x M
    Eigen::MatrixXd Normals;

    Eigen::VectorXd QDotCPlus; // Modal velocity

    ImpulseSeries Impulses;

    Eigen::Matrix3d Inertia, InvInertia;
    Eigen::Vector3d CenterOfMass;
    double Mass{0};

private:
    void CullNonSurfaceModes();
    double EffectiveMass(const Eigen::Vector3d &x, const Eigen::Vector3d &n) const;
    double EstimateContactTimeScale(int vertex, double contact_speed, const Eigen::Vector3d &impulse) const;
    void ComputeInertia();
    void LoadImpulses();

    double Dt{0};
    std::string DataPrefix;
    std::string MaterialName{"ABS Plastic"};

    Eigen::VectorXd EigenValues; // From modal analysis (omega^2 * density)
    Eigen::MatrixXd EigenVectors; // 3 * N_V x M before culling, 3 * N_S x M after

    Eigen::Matrix<double, Eigen::Dynamic, 3> V;
    Eigen::Matrix<int, Eigen::Dynamic, 3> F;

    Eigen::VectorXd Curvature;

    Eigen::VectorXd QPrev, QCur, QNext, QNextNext; // Displacement at t-1, t, t+1, t+2

    // Per-mode IIR filter coefficients, fixed for the run (computed in the constructor)
    Eigen::VectorXd CoeffQNew, CoeffQOld, CoeffQ;

    Eigen::VectorXd Force; // Modal-space force scratch, reused across Steps
    std::vector<ImpactRecord> ActiveScratch; // GetForces output scratch, reused across Steps
};

// Rigid-body acceleration noise from one impact, in the body's rest frame: the translational
// term plus the rotational one (Eq. 13 of [Chadwick et al. 2012]), both scaled by the
// half-sine contact profile. Zero outside the impact's support. Both modal paths sum this
// over their impacts, and the cross-solver comparison depends on the two agreeing.
inline Eigen::Vector3d AccelPulse(const Solver &solver, const ImpactRecord &impulse, double time) {
    if (impulse.SupportLength < 1e-12) return Eigen::Vector3d::Zero();
    const double s = ContactProfile(impulse, time);
    if (s == 0.) return Eigen::Vector3d::Zero();
    const Eigen::Vector3d r = impulse.ImpactPosition - solver.CenterOfMass;
    const double scale = std::numbers::pi * s / (2. * impulse.SupportLength);
    return impulse.ImpactVector * (scale / solver.Mass) + (solver.InvInertia * r.cross(impulse.ImpactVector) * scale).cross(r);
}

} // namespace ModalSound
