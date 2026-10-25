#include "Elastic.h"
#include <array>
#include <complex>
#include <numbers>

// Minimal material interface consumed by the unchanged SeisSol fitting routine.
namespace seissol::model {
template<std::size_t N> struct ViscoElasticMaterial {
    double lambda{1}, mu{1}, qp{30}, qs{20};
    std::array<double, N> omega;
    std::array<std::array<double, 3>, N> theta;
};
} // namespace seissol::model
#include "Equations/viscoelastic/Model/Attenuation.h"

// NOLINTNEXTLINE(misc-use-internal-linkage): Called from ElasticTest.cpp.
void CheckViscoelasticReference(elastic::Json &report) {
    using namespace elastic;
    using C = std::complex<double>;
    seissol::model::ViscoElasticMaterial<3> reference;
    seissol::physics::fitAttenuation(reference, 20., 100.);
    const double bulk = reference.lambda + 2 * reference.mu / 3;
    const double young = reference.mu * (3 * reference.lambda + 2 * reference.mu) / (reference.lambda + reference.mu);
    const double poisson = reference.lambda / (2 * (reference.lambda + reference.mu));
    Json material = {{"type", "solid"}, {"density", 1.}, {"young_modulus", young}, {"poisson_ratio", poisson}, {"relaxation", Json::array()}};
    for (int l = 0; l < 3; ++l) {
        const auto &theta = reference.theta[l];
        const double shearDecrement = -.5 * theta[2];
        const double bulkDecrement = -theta[0] - 4 * shearDecrement / 3;
        const double time = 1 / reference.omega[l];
        material["relaxation"].push_back({{"stress_time", time}, {"strain_time", {time / (1 - bulkDecrement / bulk), time / (1 - shearDecrement / reference.mu), time / (1 - shearDecrement / reference.mu), time / (1 - shearDecrement / reference.mu)}}});
    }
    const Material native = EvaluateMaterial(material, V3::Zero());
    double maximum = 0;
    for (const double frequency : {0., .2, 2., 20., 200., 2000.}) {
        const C iw(0, 2 * std::numbers::pi * frequency);
        Eigen::MatrixXcd actual = native.Stiffness.cast<C>();
        for (int i = 0; i < native.RelaxationRates.size(); ++i)
            actual -= (native.RelaxationRates[i] / (native.RelaxationRates[i] + iw)) * (native.RelaxationCoupling.col(i) * native.RelaxationCoupling.col(i).transpose()).cast<C>();
        C p = reference.lambda + 2 * reference.mu, mu = reference.mu;
        for (int i = 0; i < 3; ++i) {
            const C response = reference.omega[i] / (reference.omega[i] + iw);
            p += response * reference.theta[i][0];
            mu += response * (.5 * reference.theta[i][2]);
        }
        Eigen::MatrixXcd expected = Eigen::MatrixXcd::Zero(6, 6);
        expected.topLeftCorner(3, 3).setConstant(p - 2. * mu);
        expected.topLeftCorner(3, 3).diagonal().array() += 2. * mu;
        expected.bottomRightCorner(3, 3).diagonal().setConstant(mu);
        maximum = std::max(maximum, (actual - expected).norm() / expected.norm());
    }
    if (maximum > 1e-12) throw std::runtime_error("SeisSol complex-modulus comparison failed");
    report["seissol_attenuation"] = {{"relative_error", maximum}, {"mechanisms", 3}, {"material", material}, {"reference", "external/seissol/src/Equations/viscoelastic/Model/Attenuation.h"}, {"scope", "Unmodified attenuation fit and independent frequency response; not a SeisSol propagation run"}};
}
