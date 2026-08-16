// (c) 2024 Kangrui Xue, (c) 2023 Jui-Hsien Wang. Adapted from ModalSound / openpbso (MIT) — see NOTICE.md.
//
// Based on code by Jui-Hsien Wang (https://github.com/jhwang7628/openpbso)

#include "ModalSound.h"

#include "Mesh.h"
#include "ModeData.h"

#include <map>
#include <stdexcept>

namespace ModalSound {

namespace {
// Default ABS plastic.
struct ModalMaterial {
    int Id{0};
    double Alpha{30.};
    double Beta{1e-6};
    double Density{1070.}; // Assumed constant and uniform
    double YoungsModulus{1.4e9};
    double PoissonRatio{0.35};

    double OneMinusNu2OverE() const { return (1. - std::pow(PoissonRatio, 2)) / YoungsModulus; }
    double Xi(double omega) const { return 0.5 * (Alpha / omega + Beta * omega); } // Eq. 10, xi in [0, 1]
    double OmegaD(double omega) const { return omega * std::sqrt(1. - std::pow(Xi(omega), 2)); } // Eq. 12
};

// TODO: support other materials
ModalMaterial MaterialByName(const std::string &name) {
    if (name != "Glass") return {};
    return {.Id = 1, .Alpha = 1., .Beta = 1e-7, .Density = 2600., .YoungsModulus = 6.2e10, .PoissonRatio = 0.2};
}
} // namespace

Solver::Solver(const std::string &data_prefix, const std::string &mesh_file, const std::string &material_name, double dt)
    : Dt(dt), DataPrefix(data_prefix), MaterialName(material_name) {
    ModeData mode_data;
    mode_data.Read(data_prefix + ".modes");

    const int n_modes = mode_data.OmegaSquared.size();
    const int n_dof = n_modes > 0 ? mode_data.Modes[0].size() : 0;

    EigenValues.resize(n_modes);
    EigenVectors.resize(n_dof, n_modes);
    for (int m = 0; m < n_modes; ++m) {
        EigenValues(m) = mode_data.OmegaSquared.at(m);
        for (int d = 0; d < n_dof; ++d) EigenVectors(d, m) = mode_data.Modes.at(m).at(d);
    }
    CullNonSurfaceModes();

    ReadObj(mesh_file, V, F);
    Curvature = MeanCurvature(V, F);

    ComputeInertia();
    LoadImpulses();

    QPrev.setZero(n_modes);
    QCur.setZero(n_modes);
    QNext.setZero(n_modes);
    QNextNext.setZero(n_modes);

    QDotCPlus.setZero(n_modes);

    // The IIR filter coefficients depend only on the eigenvalues, material, and timestep,
    // all fixed for the run, so compute them once here instead of every Step.
    const auto material = MaterialByName(MaterialName);
    CoeffQNew.resize(n_modes);
    CoeffQOld.resize(n_modes);
    CoeffQ.resize(n_modes);
    for (int i = 0; i < n_modes; ++i) {
        const double omega = std::sqrt(EigenValues[i] / material.Density);
        const double omega_d = material.OmegaD(omega);

        const double xi = material.Xi(omega);
        const double epsilon = std::exp(-xi * omega * Dt);
        const double theta = omega_d * Dt;
        const double gamma = std::asin(xi);

        CoeffQNew[i] = 2. * epsilon * std::cos(theta);
        CoeffQOld[i] = epsilon * epsilon;
        CoeffQ[i] = 2. / (3. * omega * omega_d) * (epsilon * std::cos(theta + gamma) - epsilon * epsilon * std::cos(2. * theta + gamma)) / material.Density;
    }

    Force.setZero(n_modes);
}

void Solver::Step(double time) {
    Impulses.GetForces(time, ActiveScratch);

    // Force in modal space: for each impact within this timestep, add forces to the corresponding modes
    Force.setZero();
    for (const auto &impact : ActiveScratch) {
        const Eigen::Vector3d &j = impact.ImpactVector;
        const int vertex = impact.AppliedVertex;
        Force += EigenVectors.row(3 * vertex + 0) * j[0] + EigenVectors.row(3 * vertex + 1) * j[1] + EigenVectors.row(3 * vertex + 2) * j[2];
    }

    // Step the system
    QNextNext = CoeffQNew.cwiseProduct(QNext) - CoeffQOld.cwiseProduct(QCur) + CoeffQ.cwiseProduct(Force);

    QDotCPlus = (QNext - QCur) / Dt;

    // Rotate the displacement history without copying (QNextNext becomes the write scratch)
    QPrev.swap(QCur);
    QCur.swap(QNext);
    QNext.swap(QNextNext);
}

void Solver::CullNonSurfaceModes() {
    std::ifstream geo_file{DataPrefix + ".geo.txt"};
    int map_size;
    geo_file >> map_size;

    int tet_idx, surf_idx;
    double nx, ny, nz, area;
    std::map<int, int> surf_of_tet;
    std::vector<Eigen::Vector3d> normals;
    while (geo_file >> tet_idx >> surf_idx >> nx >> ny >> nz >> area) {
        surf_of_tet.insert({tet_idx, surf_idx});
        normals.emplace_back(nx, ny, nz);
    }

    const int n_surface_vertices = surf_of_tet.size();
    const int n_volume_vertices = EigenVectors.rows() / 3;
    const int n_modes = EigenVectors.cols();

    Eigen::MatrixXd culled(n_surface_vertices * 3, n_modes);
    EigenVectorsNormal.resize(n_surface_vertices, n_modes);
    Normals.resize(n_surface_vertices, 3);
    for (int vol = 0; vol < n_volume_vertices; ++vol) {
        const auto it = surf_of_tet.find(vol);
        if (it == surf_of_tet.end()) continue;

        const int surf = it->second;
        Normals.row(surf) = normals[surf];

        culled.row(surf * 3 + 0) = EigenVectors.row(vol * 3 + 0);
        culled.row(surf * 3 + 1) = EigenVectors.row(vol * 3 + 1);
        culled.row(surf * 3 + 2) = EigenVectors.row(vol * 3 + 2);
        EigenVectorsNormal.row(surf) = EigenVectors.row(vol * 3 + 0) * normals[surf][0] + EigenVectors.row(vol * 3 + 1) * normals[surf][1] + EigenVectors.row(vol * 3 + 2) * normals[surf][2];
    }

    EigenVectors = culled;
}

double Solver::EffectiveMass(const Eigen::Vector3d &x, const Eigen::Vector3d &n) const {
    const Eigen::Vector3d r = x - CenterOfMass; // Mass center = volume center
    const Eigen::Vector3d r_cross_n = r.cross(n.normalized());
    const Eigen::Vector3d premult = InvInertia * r_cross_n;
    const double one_over_m_corr = r_cross_n.dot(premult);
    return 1. / (1. / Mass + one_over_m_corr);
}

double Solver::EstimateContactTimeScale(int vertex, double contact_speed, const Eigen::Vector3d &impulse) const {
    const ModalMaterial material;
    const Eigen::Vector3d x = V.row(vertex);
    const double m = EffectiveMass(x, impulse);
    const double one_over_r = Curvature(vertex);
    const double one_over_e = material.OneMinusNu2OverE();

    return 2.87 * std::pow(std::pow(m * one_over_e, 2) * std::fabs(one_over_r / contact_speed), 0.2);
}

void Solver::LoadImpulses() {
    std::ifstream in{DataPrefix + ".impulses.txt"};

    int object_id = -1;
    char pair_order, impulse_type;
    ImpactRecord record;
    while (in >> record.Timestamp >> object_id >> record.AppliedVertex >> record.ContactSpeed >> record.ImpactVector[0] >> record.ImpactVector[1] >> record.ImpactVector[2] >> pair_order >> impulse_type) {
        if (object_id != 0) continue;

        // Was `"..." + impulseType`, which offsets the literal by the char's value and reads out of bounds.
        if (impulse_type != 'C') throw std::runtime_error(std::string{"Unsupported impulse type: "} + impulse_type);

        record.SupportLength = EstimateContactTimeScale(record.AppliedVertex, record.ContactSpeed, record.ImpactVector); // TODO: factor out
        record.ImpactPosition = V.row(record.AppliedVertex);
        record.Gamma = std::numbers::pi * record.ImpactVector.norm() / 2. / record.SupportLength;
        Impulses.Add(record); // Assumes a point impulse
    }
    Impulses.Filter();
}

void Solver::ComputeInertia() {
    std::ifstream in{DataPrefix + ".tet", std::ios::in | std::ios::binary};

    int n_verts, n_tets;
    Eigen::MatrixXd v;
    Eigen::MatrixXi t;

    Eigen::Vector3d v_buf;
    Eigen::Vector4i t_buf;
    in.read(reinterpret_cast<char *>(&n_verts), sizeof(int)); // First 4 bytes empty: skip ahead
    in.read(reinterpret_cast<char *>(&n_verts), sizeof(int));
    v.resize(n_verts, 3);
    for (int i = 0; i < n_verts; ++i) {
        in.read(reinterpret_cast<char *>(v_buf.data()), 3 * sizeof(double));
        v.row(i) = v_buf;
    }
    in.read(reinterpret_cast<char *>(&n_tets), sizeof(int));
    t.resize(n_tets, 4);
    for (int i = 0; i < n_tets; ++i) {
        in.read(reinterpret_cast<char *>(t_buf.data()), 4 * sizeof(int));
        t.row(i) = t_buf;
    }

    Eigen::VectorXd volumes = Eigen::VectorXd::Zero(n_verts); // Scale by density to get mass
    for (int i = 0; i < n_tets; ++i) {
        const Eigen::Vector3d p1 = v.row(t(i, 0)), p2 = v.row(t(i, 1)), p3 = v.row(t(i, 2)), p4 = v.row(t(i, 3));
        const Eigen::Vector3d ad = p2 - p1, bd = p3 - p1, cd = p4 - p1;
        const double vol = (std::abs(ad.dot(bd.cross(cd))) / 6.) * 0.25;

        for (int c = 0; c < 4; ++c) volumes[t(i, c)] += vol;
    }

    double total_volume = 0.;
    CenterOfMass = Eigen::Vector3d::Zero();
    for (int i = 0; i < n_verts; ++i) {
        total_volume += volumes[i];
        CenterOfMass += volumes[i] * v.row(i);
    }
    CenterOfMass /= total_volume;

    Inertia = Eigen::Matrix3d::Zero();
    for (int i = 0; i < n_verts; ++i) {
        Eigen::Vector3d ri = v.row(i);
        ri -= CenterOfMass;
        Eigen::Matrix3d tmp;
        tmp << volumes[i] * (ri[1] * ri[1] + ri[2] * ri[2]), -volumes[i] * ri[0] * ri[1], -volumes[i] * ri[0] * ri[2],
            -volumes[i] * ri[1] * ri[0], volumes[i] * (ri[0] * ri[0] + ri[2] * ri[2]), -volumes[i] * ri[1] * ri[2],
            -volumes[i] * ri[2] * ri[0], -volumes[i] * ri[2] * ri[1], volumes[i] * (ri[0] * ri[0] + ri[1] * ri[1]);
        Inertia += tmp;
    }

    const ModalMaterial material;
    InvInertia = (Inertia * material.Density).inverse();
    Mass = total_volume * material.Density;
}

} // namespace ModalSound
