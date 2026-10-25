#include "ElasticRun.h"
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <numbers>
#include <random>

using namespace elastic;
void CheckViscoelasticReference(Json &);
namespace {
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void CheckCfl(const Model &model, Json &report) {
    Mat a(model.System);
    for (const auto &e : model.Elements) {
        Eigen::LLT<Mat> const factor(e.EnergyMass);
        const Mat u = factor.matrixU(), inverse = factor.matrixU().solve(Mat::Identity(u.rows(), u.cols()));
        a.middleRows(e.Offset, u.rows()) = u * a.middleRows(e.Offset, u.rows());
        a.middleCols(e.Offset, u.rows()) = a.middleCols(e.Offset, u.rows()) * inverse;
    }
    const Eigen::EigenSolver<Mat> eigen(a, false);
    Check(eigen.info() == Eigen::Success, "CFL spectrum computation");
    const double radius = std::hypot(model.CflRealBound, model.CflImaginaryBound);
    double amplification = 0;
    for (const auto value : eigen.eigenvalues()) {
        Check(value.real() <= 1e-10 * radius && -value.real() <= model.CflRealBound * (1 + 1e-10), "CFL real spectral bound");
        Check(std::abs(value.imag()) <= model.CflImaginaryBound * (1 + 1e-10), "CFL imaginary spectral bound");
        std::complex<double> q = 1, r = 0;
        for (int stage = 0; stage < 5; ++stage) {
            r = RkA[stage] * r + model.DtLimit * value * q;
            q += RkB[stage] * r;
        }
        amplification = std::max(amplification, std::abs(q));
    }
    Check(amplification <= 1 + 1e-10, "CFL RK spectral stability");
    report.push_back({{"size", model.Size}, {"penalty", model.Penalty}, {"dt", model.DtLimit}, {"real_bound", model.CflRealBound}, {"imaginary_bound", model.CflImaginaryBound}, {"maximum_amplification", amplification}});
}
Json Fluid() { return {{"type", "fluid"}, {"density", 1.}, {"sound_speed", 1.}}; }
Json Solid() { return {{"type", "solid"}, {"density", 1.}, {"young_modulus", 2.5}, {"poisson_ratio", .25}}; }
Json Rows(const Mat &matrix) {
    Json rows = Json::array();
    for (int i = 0; i < matrix.rows(); ++i) {
        Json row = Json::array();
        for (int j = 0; j < matrix.cols(); ++j) row.push_back(matrix(i, j));
        rows.push_back(row);
    }
    return rows;
}
Json Curved(Json mesh) {
    const std::array<std::array<int, 2>, 6> edges{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    Json reference = mesh["reference_nodes"];
    std::map<std::array<int, 2>, int> nodes;
    for (auto edge : edges) {
        Json p = Json::array();
        for (int i = 0; i < 3; ++i) p.push_back(.5 * (double(reference[edge[0]][i]) + double(reference[edge[1]][i])));
        mesh["reference_nodes"].push_back(p);
    }
    for (auto &element : mesh["elements"])
        for (auto edge : edges) {
            std::array<int, 2> key{element["nodes"][edge[0]], element["nodes"][edge[1]]};
            std::sort(key.begin(), key.end());
            if (!nodes.contains(key)) {
                nodes[key] = mesh["nodes"].size();
                Json p = Json::array();
                for (int i = 0; i < 3; ++i) p.push_back(.5 * (double(mesh["nodes"][key[0]][i]) + double(mesh["nodes"][key[1]][i])));
                mesh["nodes"].push_back(p);
            }
            element["nodes"].push_back(nodes[key]);
        }
    for (auto &p : mesh["nodes"]) p[0] = double(p[0]) + .1 * double(p[1]) * double(p[2]);
    mesh["geometry_degree"] = 2;
    return mesh;
}
Json BoxMesh(int divisions, const V3 &extent, const std::function<int(const V3 &)> &tag) {
    Check(divisions > 0 && divisions <= 32 && extent.allFinite() && extent.minCoeff() > 0, "Invalid box mesh");
    Json mesh = {{"geometry_degree", 1}, {"reference_nodes", {{-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}}, {"nodes", Json::array()}, {"elements", Json::array()}};
    const auto Point = [](const Json &j) { return V3(j[0], j[1], j[2]); };
    const auto id = [&](int x, int y, int z) { return x + (divisions + 1) * (y + (divisions + 1) * z); };
    for (int z = 0; z <= divisions; ++z)
        for (int y = 0; y <= divisions; ++y)
            for (int x = 0; x <= divisions; ++x) mesh["nodes"].push_back({extent[0] * x / divisions, extent[1] * y / divisions, extent[2] * z / divisions});
    for (int z = 0; z < divisions; ++z)
        for (int y = 0; y < divisions; ++y)
            for (int x = 0; x < divisions; ++x) {
                std::array<int, 3> perm{0, 1, 2};
                do {
                    std::array<int, 3> at{x, y, z};
                    std::array<int, 4> vertices{id(x, y, z), 0, 0, id(x + 1, y + 1, z + 1)};
                    ++at[perm[0]];
                    vertices[1] = id(at[0], at[1], at[2]);
                    ++at[perm[1]];
                    vertices[2] = id(at[0], at[1], at[2]);
                    const V3 p0 = Point(mesh["nodes"][vertices[0]]);
                    Eigen::Matrix3d j;
                    V3 center = p0;
                    for (int k = 0; k < 3; ++k) {
                        const V3 p = Point(mesh["nodes"][vertices[k + 1]]);
                        j.col(k) = p - p0;
                        center += p;
                    }
                    if (j.determinant() < 0) std::swap(vertices[1], vertices[2]);
                    mesh["elements"].push_back({{"tag", tag(center / 4)}, {"nodes", vertices}});
                } while (std::next_permutation(perm.begin(), perm.end()));
            }
    return mesh;
}
double Evolution(Model &model, const std::function<Vec(const V3 &, int, double)> &exact, double duration, Json *regions = nullptr) {
    model.Initial = model.Project([&](const V3 &x, int tag) { return exact(x, tag, 0); });
    model.BoundaryState = exact;
    const auto trace = model.RunReference(duration, 1 / duration);
    const Vec reference = model.Project([&](const V3 &x, int tag) { return exact(x, tag, duration); });
    if (regions) {
        std::map<int, std::array<double, 2>> errors;
        for (const auto &e : model.Elements) {
            const auto r = reference.segment(e.Offset, model.Nodes * e.Fields());
            const Vec d = trace.State.segment(e.Offset, model.Nodes * e.Fields()) - r;
            errors[e.Tag][0] += d.dot(e.PhysicalMass * d);
            errors[e.Tag][1] += r.dot(e.PhysicalMass * r);
        }
        for (const auto &[tag, error] : errors) (*regions)[std::to_string(tag)] = std::sqrt(error[0] / error[1]);
    }
    return std::sqrt(model.Energy(trace.State - reference, true) / model.Energy(reference, true));
}
double ScholteSpeed() {
    const auto residual = [](double c) { const double a=std::sqrt(1-c*c),p=std::sqrt(1-c*c/3),s=std::sqrt(1-c*c); return c*c*p*(1-s*s)/(a*(s*s+1))+(p*p-1)+2*p*p-4*s*p/(s*s+1); };
    double lo = .1, hi = .99;
    Check(residual(lo) * residual(hi) < 0, "Scholte root bracket");
    for (int i = 0; i < 60; ++i) {
        const double middle = .5 * (lo + hi);
        if (residual(lo) * residual(middle) > 0) lo = middle;
        else hi = middle;
    }
    return .5 * (lo + hi);
}
Vec Scholte(const V3 &x, int tag, double time) {
    using C = std::complex<double>;
    const C imaginary(0, 1);
    static const double c = ScholteSpeed();
    const double k = 2 / c, w = 2, qa = k * std::sqrt(1 - c * c), qp = k * std::sqrt(1 - c * c / 3), qs = k * std::sqrt(1 - c * c);
    const C b = 1., d = -2. * imaginary * k * qp / (qs * qs + k * k), a = qp * (k * k - qs * qs) / (qa * (qs * qs + k * k));
    const double y = x[1] - .5;
    const C phase = std::exp(imaginary * (k * x[0] - w * time));
    if (tag == 1) {
        const C phi = a * std::exp(-qa * y) * phase;
        Vec q = Vec::Zero(4);
        q[0] = std::real(w * k * phi);
        q[1] = std::real(imaginary * w * qa * phi);
        q[3] = std::real(-w * w * phi);
        return q;
    }
    const C p = b * std::exp(qp * y) * phase, s = d * std::exp(qs * y) * phase;
    Vec q = Vec::Zero(9);
    q[0] = std::real(-imaginary * w * (imaginary * k * p + qs * s));
    q[1] = std::real(-imaginary * w * (qp * p - imaginary * k * s));
    const C divergence = (qp * qp - k * k) * p;
    q[3] = std::real(divergence - 2 * k * k * p + 2. * imaginary * k * qs * s);
    q[4] = std::real(divergence + 2 * qp * qp * p - 2. * imaginary * k * qs * s);
    q[5] = std::real(divergence);
    q[8] = std::real(2. * imaginary * k * qp * p + (qs * qs + k * k) * s);
    return q;
}

void ViscoelasticChecks(Json &report) {
    CheckViscoelasticReference(report);
    Json material = Solid();
    material["relaxation"] = {{{"stress_time", {.05, .04, .03, .02}}, {"strain_time", {.07, .064, .054, .04}}}};
    const auto m = EvaluateMaterial(material, V3::Zero());
    Check(m.Fields() == 15 && m.RelaxationRates.size() == 6, "Viscoelastic state layout");
    for (const auto &times : {Json{{"stress_time", 0}, {"strain_time", .1}}, Json{{"stress_time", .1}, {"strain_time", .05}}}) {
        bool rejected = false;
        try {
            auto bad = material;
            bad["relaxation"] = {times};
            EvaluateMaterial(bad, V3::Zero());
        } catch (const std::invalid_argument &) { rejected = true; }
        Check(rejected, "Invalid relaxation times accepted");
    }
    const auto mesh = BoxMesh(1, V3::Ones(), [](const V3 &) { return 2; });
    {
        const Json config = {{"degree", 1}, {"mass", "full"}, {"materials", {{"2", material}}}, {"default_boundary", "fixed"}};
        Model model(config, mesh);
        const Vec strain = Vec::LinSpaced(6, .01, .06);
        const auto exact = [&](double time) {
            Vec q = Vec::Zero(m.Fields());
            const Vec initial = m.RelaxationCoupling.transpose() * strain;
            q.tail(initial.size()) = (-time * m.RelaxationRates.array()).exp().matrix().cwiseProduct(initial);
            q.segment(3, 6) = m.Stiffness * strain + m.RelaxationCoupling * (q.tail(initial.size()) - initial);
            return q;
        };
        model.Initial = model.Project([&](const V3 &, int) { return exact(0); });
        const Vec final = model.Project([&](const V3 &, int) { return exact(.1); });
        const auto cpu = model.Run(.1, 1000, RunOptions{true, .001});
        const auto gpu = model.Run(.1, 1000, RunOptions{false, .001});
        const double error = std::sqrt(model.Energy(cpu.State - final) / model.Energy(final));
        Check(error < 1e-8, "Analytic stress relaxation");
        Check(std::sqrt(model.Energy(gpu.State - cpu.State) / model.Energy(cpu.State)) < 1e-9, "Metal stress relaxation");
        Check(gpu.Energy(gpu.Energy.rows() - 1, 0) < model.Energy(model.Initial), "Relaxation energy decay");
        report["stress_relaxation_error"] = error;
    }
    for (const bool shear : {false, true}) {
        using C = std::complex<double>;
        const C imaginary(0, 1);
        const double omega = 2;
        // Independent frequency-domain Zener moduli, normalized by their high-frequency values.
        const auto modulus = [&](int mode) {
            const double ts = material["relaxation"][0]["stress_time"][mode];
            const double te = material["relaxation"][0]["strain_time"][mode];
            return (ts / te) * (1. + imaginary * omega * te) / (1. + imaginary * omega * ts);
        };
        const C bulk = (5. / 3) * modulus(0), mu = modulus(shear ? 3 : 1);
        const C longitudinal = bulk + (4. / 3) * mu, speed = std::sqrt((shear ? mu : longitudinal) / m.Density);
        const C k = omega / speed;
        Check(k.imag() < 0, "Passive wave attenuation sign");
        Eigen::Matrix<C, 6, 6> stiffness = m.Stiffness.cast<C>();
        for (int j = 0; j < m.RelaxationRates.size(); ++j)
            stiffness -= (m.RelaxationRates[j] / (m.RelaxationRates[j] + imaginary * omega)) * (m.RelaxationCoupling.col(j) * m.RelaxationCoupling.col(j).transpose()).cast<C>();
        Check(std::abs(stiffness(shear ? 5 : 0, shear ? 5 : 0) - (shear ? mu : longitudinal)) < 1e-13, "Paper relaxation versus Zener complex modulus");
        const V3 velocity = shear ? V3::UnitY() : V3::UnitX();
        const Vec strain = NormalMatrix(V3::UnitX()) * velocity;
        Eigen::VectorXcd amplitude(m.Fields());
        amplitude.head(3) = velocity.cast<C>();
        amplitude.segment(3, 6) = -(k / omega) * stiffness * strain.cast<C>();
        for (int j = 0; j < m.RelaxationRates.size(); ++j)
            amplitude[9 + j] = -imaginary * k * m.RelaxationCoupling.col(j).dot(strain) / (m.RelaxationRates[j] + imaginary * omega);
        Check((omega * m.Density * amplitude.head(3) + k * NormalMatrix(V3::UnitX()).transpose().cast<C>() * amplitude.segment(3, 6)).norm() < 1e-12, "Analytic momentum balance");
        const auto exact = [&](const V3 &x, int, double time) -> Vec { return (amplitude * std::exp(imaginary * (omega * time - k * x[0]))).real(); };
        double previous = 1;
        for (int degree : {1, 2, 3, 4}) {
            Json const config = {{"degree", degree}, {"mass", "weight_adjusted"}, {"materials", {{"2", material}}}, {"default_boundary", "prescribed"}};
            Model model(config, mesh);
            const double error = Evolution(model, exact, .04);
            std::cout << "viscoelastic " << (shear ? "S" : "P") << " degree " << degree << " error " << error << std::endl;
            Check(error < previous * .8, "Viscoelastic plane-wave convergence");
            previous = error;
            report[shear ? "viscoelastic_s_wave" : "viscoelastic_p_wave"].push_back(error);
        }
        Check(previous < .002, "Viscoelastic plane-wave accuracy");
    }
    for (const auto *mass : {"full", "weight_adjusted"}) {
        Json varying = material;
        const auto elastic = EvaluateMaterial(Solid(), V3::Zero());
        Mat stiffness = elastic.Stiffness;
        stiffness(0, 0) *= 1.2;
        stiffness(0, 5) = stiffness(5, 0) = .1;
        varying["stiffness"] = Rows(stiffness);
        varying["density"] = {{"value", 1.}, {"gradient", {.1, .02, 0.}}};
        varying["relaxation"].push_back({{"stress_time", .08}, {"strain_time", .088}});
        varying["relaxation"][0]["stress_time"][0] = {{"value", .05}, {"gradient", {.001, 0., 0.}}};
        const Json config = {{"degree", 1}, {"mass", mass}, {"materials", {{"1", Fluid()}, {"2", varying}, {"3", Solid()}}}, {"default_boundary", "absorbing"}};
        const auto mixed = Curved(BoxMesh(1, V3::Ones(), [](const V3 &x) { return x[0] < .4 ? 1 : x[0] > .6 ? 3 :
                                                                                                              2; }));
        Model model(config, mixed);
        model.Initial = model.Project([&](const V3 &x, int tag) {
            const auto material = EvaluateMaterial(config["materials"][std::to_string(tag)], x);
            Vec q = Vec::Zero(material.Fields());
            q.head(3) << .1 * x[0], .2 * x[1], -.1 * x[2];
            if (material.Fluid()) q[3] = .3;
            else q.segment(3, 3).setConstant(.3);
            if (tag == 2) q[9] = .02;
            return q;
        });
        const Vec rhs = model.Apply(model.Initial);
        Check((rhs - model.ApplyMatrixFree(model.Initial)).norm() < 1e-11 * rhs.norm(), "Viscoelastic matrix-free equivalence");
        const auto d = model.Diagnostics(model.Initial, 0);
        Check(d[5] > 0 && d[4] >= -1e-12, "Separate material and interface dissipation");
        Check(std::abs(model.Power(model.Initial, rhs) - (d[2] + d[3] - d[4] - d[5])) < 1e-11, "Viscoelastic energy balance");
        CheckCfl(model, report["viscoelastic_cfl"]);
        const auto cpu = model.Run(.02, 200, RunOptions{true, 0});
        Model local(config, mixed, false);
        local.Initial = model.Initial;
        const auto gpu = local.Run(.02, 200);
        const double stateError = std::sqrt(model.Energy(gpu.State - cpu.State) / model.Energy(cpu.State));
        const double diagnosticError = (gpu.Energy - cpu.Energy).norm() / cpu.Energy.norm();
        Check(stateError < 1e-8 && diagnosticError < 1e-8, "Viscoelastic Metal agreement");
        Check(gpu.Energy.col(5).minCoeff() >= 0 && gpu.Energy(gpu.Energy.rows() - 1, 0) < model.Energy(model.Initial), "Viscoelastic decay");
        report["viscoelastic_backends"][mass] = {{"state_error", stateError}, {"diagnostic_error", diagnosticError}};
    }
    Json zero = Solid();
    zero["relaxation"] = {{{"stress_time", .05}, {"strain_time", .05}}};
    const Json config = {{"degree", 1}, {"materials", {{"2", Solid()}}}, {"default_boundary", "absorbing"}};
    Model lossless(config, mesh);
    auto zeroConfig = config;
    zeroConfig["materials"]["2"] = zero;
    Model limit(zeroConfig, mesh);
    const auto initial = [](const V3 &x, int) { Vec q = Vec::Zero(9); q[0] = x[0]; q[3] = .2; return q; };
    lossless.Initial = lossless.Project(initial);
    limit.Initial = limit.Project([&](const V3 &x, int tag) { Vec q = Vec::Zero(15); q.head(9) = initial(x, tag); return q; });
    const Vec a = lossless.Apply(lossless.Initial), b = limit.Apply(limit.Initial);
    for (int i = 0; i < int(limit.Elements.size()); ++i) {
        Check((a.segment(lossless.Elements[i].Offset, 9 * lossless.Nodes) - b.segment(limit.Elements[i].Offset, 9 * limit.Nodes)).norm() < 1e-11, "Lossless relaxation limit");
        Check(b.segment(limit.Elements[i].Offset + 9 * limit.Nodes, 6 * limit.Nodes).norm() == 0, "Unexcited zero-strength memory");
    }
    report["viscoelastic_passed"] = true;
}

} // namespace
int main(int argc, char *const *argv) try {
    Json report;
    if (argc > 1 && std::string(argv[1]) == "--viscoelastic") {
        ViscoelasticChecks(report);
        std::cout << report.dump(2) << '\n';
        return 0;
    }
    const auto fluid = EvaluateMaterial(Fluid(), V3::Zero()), solid = EvaluateMaterial(Solid(), V3::Zero());
    const V3 n = V3(1, 2, 3).normalized();
    Vec f(4), s(9);
    f << .3, .2, .1, 2;
    s.setZero();
    s.head(3) = f.head(3);
    s.segment(3, 3).setConstant(2);
    const Vec ff = Flux(f, s, fluid, solid, n, 1), sf = Flux(s, f, solid, fluid, n, 1);
    Check((ff.head<3>() - 2 * n).norm() < 1e-14 && std::abs(ff[3] - n.dot(f.head<3>())) < 1e-14, "Coupled consistent fluid flux");
    Check((sf.head<3>() - 2 * n).norm() < 1e-14 && (sf.tail(6) - NormalMatrix(n) * s.head(3)).norm() < 1e-14, "Coupled consistent solid flux");
    std::mt19937 rng(7);
    std::normal_distribution<double> normal;
    for (int i = 0; i < 30; ++i) {
        for (auto *q : {&f, &s})
            for (int j = 0; j < q->size(); ++j) (*q)[j] = normal(rng);
        const double energy = f.dot(Flux(f, s, fluid, solid, n, 1)) - f[3] * n.dot(f.head<3>()) + s.dot(Flux(s, f, solid, fluid, -n, 1)) + s.head<3>().dot(NormalMatrix(n).transpose() * s.tail(6));
        Check(energy <= 1e-12, "Coupled interface energy production");
    }
    for (const bool coupled : {false, true})
        for (const auto *mass : {"full", "weight_adjusted"}) {
            Json const config = {{"degree", 1}, {"mass", mass}, {"materials", {{"1", Fluid()}, {"2", Solid()}}}, {"default_boundary", "absorbing"}};
            const auto mesh = BoxMesh(1, V3::Ones(), [&](const V3 &x) { return coupled && x[0] > .5 ? 2 : 1; });
            Model model(config, mesh);
            CheckCfl(model, report["cfl_spectra"]);
            Vec q(model.Size);
            for (int i = 0; i < q.size(); ++i) q[i] = normal(rng);
            Check((model.ApplyMatrixFree(q) - model.Apply(q)).norm() < 1e-11 * model.Apply(q).norm(), "Matrix-free CPU operator equivalence");
            Check(model.Energy(q) > 0 && model.Power(q, model.Rhs(q)) < 0, "Discrete energy growth");
            const auto powers = model.Powers(q, 0);
            Check(powers[1] < 0 && powers[2] >= -1e-10, "Boundary/interface power signs");
            Check(std::abs(powers[0] + powers[1] - powers[2] - model.Power(q, model.Rhs(q))) < 1e-10, "Energy budget");
            const auto state = model.Project([&](const V3 &, int tag) { Vec value=Vec::Zero(tag==1?4:9); if(tag==1)value[3]=1; else value.segment(3,3).setConstant(1); return value; });
            Check(state.allFinite(), "Projection");
            model.Initial = q;
            const auto trace = model.RunReference(.01, 100);
            Check(model.Energy(trace.State) < model.Energy(q), "Time integration energy growth");
            std::cout << "PASS " << (coupled ? "coupled" : "fluid") << ' ' << mass << " energy " << model.Energy(q) << " -> " << model.Energy(trace.State) << '\n';
        }

    // The exact fields exercise all directions and the solid-fluid traction convention.
    const auto scholteMesh = BoxMesh(2, V3::Ones(), [](const V3 &x) { return x[1] > .5 ? 1 : 2; });
    double previous = 1;
    for (int degree : {1, 2, 3}) {
        Json const config = {{"degree", degree}, {"materials", {{"1", Fluid()}, {"2", Solid()}}}, {"default_boundary", "prescribed"}};
        Model model(config, scholteMesh);
        const double error = Evolution(model, Scholte, .02);
        std::cout << "Scholte degree " << degree << " relative energy norm error " << error << '\n';
        Check(error < previous * .85, "Scholte convergence");
        previous = error;
        report["scholte_errors"].push_back(error);
    }
    report["scholte_speed"] = ScholteSpeed();
    const V3 direction = V3(1, 2, 3).normalized();
    for (bool const solidMedium : {false, true}) {
        const Json material = solidMedium ? Solid() : Fluid();
        const auto m = EvaluateMaterial(material, V3::Zero());
        const double c = solidMedium ? std::numbers::sqrt3 : 1.;
        const auto exact = [&](const V3 &x, int, double time) -> Vec { Vec q=Vec::Zero(m.Fields()); const double value=std::sin(2*(direction.dot(x)-c*time)); q.head<3>()=direction*value; if(m.Fluid())q[3]=-m.Impedance*value; else q.tail(6)=-m.Stiffness*NormalMatrix(direction)*direction*value/c; return q; };
        double last = 1;
        for (int degree : {1, 2, 3, 4}) {
            Json const config = {{"degree", degree}, {"materials", {{"1", material}}}, {"default_boundary", "prescribed"}};
            Model model(config, Curved(BoxMesh(1, V3::Ones(), [](const V3 &) { return 1; })));
            const auto linear = [](const V3 &x, int, double) -> Vec {Vec q=Vec::Zero(4);q[3]=x[0];return q; };
            if (!solidMedium && degree >= 2) {
                model.BoundaryState = linear;
                auto const u = model.Project([&](const V3 &x, int tag) { return linear(x, tag, 0); });
                const Vec expected = model.Project([](const V3 &, int) -> Vec {Vec q=Vec::Zero(4);q[0]=1;return q; });
                Check((model.Rhs(u) - expected).norm() / expected.norm() < 1e-10, "Curved linear consistency");
            }
            const double error = Evolution(model, exact, .01);
            if (degree == 2) {
                Model fine(config, Curved(BoxMesh(2, V3::Ones(), [](const V3 &) { return 1; })));
                const double refined = Evolution(fine, exact, .01);
                std::cout << "Mesh refinement " << error << " -> " << refined << std::endl;
                Check(refined < .75 * error, "Curved mesh refinement");
                report[solidMedium ? "solid_curved_refined" : "fluid_curved_refined"] = refined;
            }
            Check(degree < 4 || error < last * .1, "Curved wave convergence");
            last = error;
            report[solidMedium ? "solid_curved_errors" : "fluid_curved_errors"].push_back(error);
            std::cout << "Curved " << solidMedium << " degree " << degree << " error " << error << '\n';
        }
    }
    // Measure the transmitted wave separately at a large impedance contrast.
    Json metal = Solid();
    metal["density"] = 1000.;
    metal["young_modulus"] = 2500.;
    const double zs = 1000 * std::numbers::sqrt3, reflection = (zs - 1) / (zs + 1), transmission = 2 * zs / (zs + 1);
    const auto normalWave = [&](const V3 &x, int tag, double time) -> Vec {
        const double incoming = std::cos(2 * (x[0] - time)), reflected = reflection * std::cos(2 * (1 - x[0] - time));
        Vec q = Vec::Zero(tag == 1 ? 4 : 9);
        if (tag == 1) {
            q[0] = incoming - reflected;
            q[3] = -incoming - reflected;
        } else {
            const double p = transmission * std::cos(2 * (.5 + (x[0] - .5) / std::numbers::sqrt3 - time));
            q[0] = p / zs;
            q[3] = -p;
            q[4] = q[5] = -p / 3;
        }
        return q;
    };
    Json const highConfig = {{"degree", 4}, {"materials", {{"1", Fluid()}, {"2", metal}}}, {"default_boundary", "prescribed"}};
    Model high(highConfig, BoxMesh(2, V3::Ones(), [](const V3 &x) { return x[0] < .5 ? 1 : 2; }));
    report["high_contrast_error"] = Evolution(high, normalWave, .01, &report["high_contrast_regions"]);
    Check(double(report["high_contrast_error"]) < .01, "High impedance contrast accuracy");
    std::cout << report["high_contrast_regions"].dump() << std::endl;
    for (const auto &error : report["high_contrast_regions"]) Check(double(error) < .01, "Transmitted wave accuracy");
    for (const auto *mass : {"full", "weight_adjusted"}) {
        Json anisotropic = Solid();
        anisotropic.erase("young_modulus");
        anisotropic.erase("poisson_ratio");
        Mat c = solid.Stiffness;
        c(0, 0) *= 1.2;
        c(1, 1) *= 1.4;
        c(3, 3) *= .8;
        c(0, 3) = c(3, 0) = .15;
        anisotropic["stiffness"] = Rows(c);
        for (auto &row : anisotropic["stiffness"])
            for (auto &value : row) value = {{"value", value}, {"gradient", {.1 * double(value), 0, 0}}};
        anisotropic["density"] = {{"value", 1.}, {"gradient", {0, .1, 0}}};
        Json const config = {{"degree", 2}, {"mass", mass}, {"materials", {{"1", anisotropic}}}, {"default_boundary", "free"}};
        Model model(config, Curved(BoxMesh(1, V3::Ones(), [](const V3 &) { return 1; })));
        Vec q(model.Size);
        for (int i = 0; i < q.size(); ++i) q[i] = normal(rng);
        const double power = model.Power(q, model.Rhs(q));
        Check(power < 0, "Anisotropic heterogeneous energy growth");
        report["anisotropic_power"][mass] = power;
        if (mass == std::string("full")) {
            model.Initial = q;
            const double dt = std::scalbn(1., std::ilogb(model.DtLimit));
            const auto cpu = model.RunReference(dt, 1 / dt, dt);
            const auto gpu = model.Run(dt, 1 / dt, RunOptions{false, dt});
            Check(std::sqrt(model.Energy(gpu.State - cpu.State) / model.Energy(cpu.State)) < 1e-11 && (gpu.Energy - cpu.Energy).norm() < 1e-11 * cpu.Energy.norm(), "Full-mass GPU state and diagnostics");
        }
    }
    {
        Json config = {{"degree", 1}, {"penalty", 0.}, {"materials", {{"1", Fluid()}}}, {"default_boundary", "rigid"}};
        Model model(config, BoxMesh(1, V3::Ones(), [](const V3 &) { return 1; }));
        for (int i = 0; i < model.Size; ++i) model.Initial[i] = normal(rng);
        Check(std::abs(model.Power(model.Initial, model.Rhs(model.Initial))) < 1e-11, "Closed central-flux energy conservation");
        const auto a = model.RunReference(.02, 50), b = model.RunReference(.02, 50, a.Dt / 2), c = model.RunReference(.02, 50, b.Dt / 2);
        const double coarse = (a.State - b.State).norm(), fine = (b.State - c.State).norm();
        std::cout << "RK differences " << coarse << " " << fine << std::endl;
        Check(coarse > 8 * fine && coarse / model.Initial.norm() < 1e-5, "RK timestep refinement");
        report["time_refinement_ratio"] = coarse / fine;
        config["degree"] = 6;
        Model high(config, BoxMesh(1, V3::Ones(), [](const V3 &) { return 1; }));
        high.Initial = high.Project([](const V3 &x, int) -> Vec { Vec q = Vec::Zero(4); q[3] = std::cos(x.sum()); return q; });
        const double dt = std::scalbn(1., std::ilogb(high.DtLimit));
        const auto cpu = high.RunReference(16 * dt, 1 / dt, dt);
        const auto gpu = high.Run(16 * dt, 1 / dt, RunOptions{false, dt});
        Check(std::sqrt(high.Energy(gpu.State - cpu.State) / high.Energy(cpu.State)) < 1e-11 && (gpu.Energy - cpu.Energy).norm() < 1e-10 * cpu.Energy.norm(), "Degree-six GPU packing, quadrature tiling and diagnostics");
    }
    {
        Json material = Solid();
        Mat stiffness = solid.Stiffness;
        stiffness(0, 0) *= 1.2;
        stiffness(1, 1) *= 1.4;
        stiffness(0, 3) = stiffness(3, 0) = .15;
        material["stiffness"] = Rows(stiffness);
        const Mat b = NormalMatrix(direction);
        Eigen::SelfAdjointEigenSolver<Mat> const christoffel(b.transpose() * stiffness * b);
        const double speed = std::sqrt(christoffel.eigenvalues()[2]);
        const V3 polarization = christoffel.eigenvectors().col(2);
        const auto exact = [&](const V3 &x, int, double t) -> Vec {const double wave=std::sin(2*(direction.dot(x)-speed*t));Vec q(9);q.head(3)=polarization*wave;q.tail(6)=-stiffness*b*polarization*wave/speed;return q; };
        Json const config = {{"degree", 3}, {"materials", {{"1", material}}}, {"default_boundary", "prescribed"}};
        Model model(config, Curved(BoxMesh(1, V3::Ones(), [](const V3 &) { return 1; })));
        const double error = Evolution(model, exact, .01);
        Check(error < .01, "Anisotropic wave accuracy");
        report["anisotropic_wave_error"] = error;
    }
    {
        Json const config = {{"degree", 1}, {"penalty", 0.}, {"materials", {{"1", Fluid()}}}, {"default_boundary", "rigid"}, {"sources", {{{"type", "volume_flow"}, {"position", {.31, .23, .19}}, {"frequency", 10.}, {"time", .08}, {"amplitude", .001}}}}, {"receivers", {{{"position", {.7, .6, .4}}, {"field", "pressure"}}}}};
        Model const model(config, BoxMesh(1, V3::Ones(), [](const V3 &) { return 1; }));
        const auto trace = model.RunReference(.2, 1000);
        const double work = (trace.Energy.col(2).sum() - .5 * trace.Energy(trace.Energy.rows() - 1, 2)) / 1000;
        const double error = std::abs(work - model.Energy(trace.State)) / model.Energy(trace.State);
        Check(error < .002 && trace.Samples.norm() > 0, "Source work and receiver output");
        report["source_work_relative_error"] = error;
        Json bad = Fluid();
        bad["sound_speed"] = -1;
        bool rejected = false;
        try {
            EvaluateMaterial(bad, V3::Zero());
        } catch (const std::invalid_argument &) { rejected = true; }
        Check(rejected, "Negative sound speed accepted");
    }
    {
        auto mesh = BoxMesh(1, V3::Ones(), [](const V3 &) { return 1; });
        mesh["boundaries"] = Json::array();
        for (const auto &element : mesh["elements"]) {
            std::vector<int> face;
            for (const auto &node : element["nodes"])
                if (double(mesh["nodes"][int(node)][0]) == 1) face.push_back(node);
            if (face.size() == 3) mesh["boundaries"].push_back({{"nodes", face}, {"tag", 10}});
        }
        const double cosine = std::sqrt(.5), reflection = (cosine - 1) / (cosine + 1);
        const V3 incident(cosine, cosine, 0), reflected(-cosine, cosine, 0);
        const auto exact = [&](const V3 &x, int, double t) -> Vec {
            const double a = std::cos(2 * (incident.dot(x) - t)), b = reflection * std::cos(2 * (2 * cosine + reflected.dot(x) - t));
            Vec q(4);
            q.head(3) = incident * a + reflected * b;
            q[3] = -a - b;
            return q;
        };
        Json const config = {{"degree", 3}, {"materials", {{"1", Fluid()}}}, {"default_boundary", "prescribed"}, {"boundaries", {{"10", "absorbing"}}}};
        Model model(config, mesh);
        const double error = Evolution(model, exact, .05);
        Check(error < .01, "Oblique characteristic boundary consistency");
        report["absorbing_oblique_error"] = error;
        report["absorbing_pressure_reflection_45deg"] = reflection;
    }
    {
        Json fluidMaterial = Fluid(), solidMaterial = Solid();
        fluidMaterial["density"] = .001;
        fluidMaterial["sound_speed"] = 10.;
        Mat stiffness = 1000 * solid.Stiffness;
        stiffness(0, 0) *= 1.3;
        stiffness(0, 3) = stiffness(3, 0) = 100;
        solidMaterial["stiffness"] = Rows(stiffness);
        solidMaterial["density"] = {{"value", 1.}, {"gradient", {0, .1, 0}}};
        Json const config = {{"degree", 2}, {"materials", {{"1", fluidMaterial}, {"2", solidMaterial}}}, {"default_boundary", "absorbing"}, {"sources", {{{"type", "volume_flow"}, {"position", {.2, .23, .19}}, {"frequency", 300.}, {"time", .001}, {"amplitude", .001}}, {{"type", "force"}, {"position", {.8, .23, .19}}, {"direction", {1, 0, 0}}, {"frequency", 200.}, {"time", .002}, {"amplitude", .01}}}}, {"receivers", {{{"position", {.2, .3, .4}}, {"field", "pressure"}}, {{"position", {.8, .3, .4}}, {"field", "vx"}}}}};
        for (double const penalty : {0., 1., 5.}) {
            auto probe = config;
            probe["penalty"] = penalty;
            probe["sources"] = probe["receivers"] = Json::array();
            CheckCfl(Model(probe, Curved(BoxMesh(1, V3::Ones(), [](const V3 &x) { return x[0] < .5 ? 1 : 2; }))), report["cfl_spectra"]);
        }
        Model model(config, Curved(BoxMesh(2, V3::Ones(), [](const V3 &x) { return x[0] < .5 ? 1 : 2; })));
        for (const auto &e : model.Elements) {
            const int n = model.Nodes * e.Fields();
            Vec q(n);
            for (int i = 0; i < n; ++i) q[i] = 1e-4 * normal(rng);
            const Eigen::LLT<Mat> factor(e.EnergyMass);
            model.Initial.segment(e.Offset, n) = factor.matrixU().solve(q);
        }
        Check((model.ApplyMatrixFree(model.Initial) - model.Apply(model.Initial)).norm() < 1e-10 * model.Apply(model.Initial).norm(), "Curved heterogeneous matrix-free CPU equivalence");
        const auto cpu = model.Run(.004, 1000, RunOptions{true});
        Model deviceModel(config, Curved(BoxMesh(2, V3::Ones(), [](const V3 &x) { return x[0] < .5 ? 1 : 2; })), false);
        deviceModel.Initial = model.Initial;
        Check(deviceModel.System.nonZeros() == 0 && deviceModel.DtLimit == model.DtLimit, "Matrix-free preparation retains CFL without reference matrix");
        const auto gpu = deviceModel.Run(.004, 1000);
        const double difference = std::sqrt(model.Energy(gpu.State - cpu.State, true) / model.Energy(cpu.State, true));
        const double diagnostics = (gpu.Energy - cpu.Energy).norm() / cpu.Energy.norm();
        std::cout << "Fixed-step GPU error " << difference << " diagnostics " << diagnostics << " steps " << cpu.Steps << "/" << gpu.Steps << std::endl;
        Check(gpu.Steps == cpu.Steps && gpu.Dt == cpu.Dt && gpu.Dt <= model.DtLimit, "Fixed timestep and backend step agreement");
        Check(difference < 1e-10 && diagnostics < 1e-10 && (gpu.Samples - cpu.Samples).norm() < 1e-10 * cpu.Samples.norm(), "GPU coupled state, receivers and diagnostics");
        report["fixed_step"] = {{"dt", gpu.Dt}, {"gpu_cpu_state_error", difference}, {"gpu_cpu_diagnostic_error", diagnostics}, {"steps", gpu.Steps}};
        const auto half = model.Run(.004, 1000, RunOptions{true, .5 * cpu.Dt});
        const auto quarter = model.Run(.004, 1000, RunOptions{true, .25 * cpu.Dt});
        const auto eighth = model.Run(.004, 1000, RunOptions{true, .125 * cpu.Dt});
        const double coarseError = std::sqrt(model.Energy(cpu.State - half.State, true));
        const double fineError = std::sqrt(model.Energy(half.State - quarter.State, true));
        const double refinedError = std::sqrt(model.Energy(quarter.State - eighth.State, true));
        std::cout << "CFL differences " << coarseError << " " << fineError << " " << refinedError << std::endl;
        Check(fineError < coarseError && refinedError < .125 * fineError, "CFL timestep refinement towards fourth-order convergence");
        report["cfl_refinement"] = {{"coarse_difference", coarseError}, {"fine_difference", fineError}, {"refined_difference", refinedError}, {"refined_ratio", fineError / refinedError}};
        const auto splitCpu = model.Run(.001, 1000, RunOptions{true, cpu.Dt / 16});
        const auto splitGpu = deviceModel.Run(.001, 1000, RunOptions{false, cpu.Dt / 16});
        Check(splitGpu.Steps > CommandSteps && splitGpu.Steps == splitCpu.Steps, "Split-command fixture");
        Check(std::sqrt(model.Energy(splitGpu.State - splitCpu.State, true) / model.Energy(splitCpu.State, true)) < 1e-10 && (splitGpu.Energy - splitCpu.Energy).norm() < 1e-10 * splitCpu.Energy.norm(), "Split-command GPU state and diagnostics");
        deviceModel.Sources.clear();
        deviceModel.Initial.setZero();
        const auto zero = deviceModel.Run(.002, 1000);
        Check(zero.State.isZero() && zero.Energy.isZero() && zero.Samples.isZero(), "Zero unforced GPU state");
    }
    ViscoelasticChecks(report);
    report["passed"] = true;
    if (argc > 1) {
        std::ofstream out(argv[1]);
        out << report.dump(2) << '\n';
        Check(bool(out), "Cannot write test report");
    }
    std::cout << "PASS elastic validation\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
