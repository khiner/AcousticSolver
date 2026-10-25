#include "Elastic.h"
#include <Eigen/SparseLU>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

using namespace elastic;
namespace {
using Complex = std::complex<double>;
Json Read(const std::filesystem::path &p) {
    std::ifstream f(p);
    return Json::parse(f);
}
Json Coordinates(const Mat &points) {
    Json out = Json::array();
    for (int q = 0; q < points.rows(); ++q) out.push_back({points(q, 0), points(q, 1), points(q, 2)});
    return out;
}
Vec Row(const Json &data, int row, int fields) {
    if (data.at(row).size() != size_t(fields)) throw std::runtime_error("Reference field count mismatch");
    Vec value(fields);
    for (int i = 0; i < fields; ++i) value[i] = data.at(row).at(i);
    if (!value.allFinite()) throw std::runtime_error("Nonfinite reference field");
    return value;
}
} // namespace
int main(int argc, char *const *argv) try {
    if (argc != 4 && argc != 5) throw std::runtime_error("Usage: ElasticReference CONFIG POINTS.json REPORT.json [FIELDS.json]");
    const std::filesystem::path path = argv[1];
    auto config = Read(path);
    config["default_boundary"] = "prescribed";
    config.erase("boundaries");
    auto mesh = Read(path.parent_path() / config.at("mesh").get<std::string>());
    mesh.erase("boundaries");
    Model model(config, mesh);
    if (!model.Sources.empty()) throw std::runtime_error("Harmonic reference inputs cannot include transient point sources");
    if (argc == 4) {
        Json points = {{"elements", Json::array()}, {"boundaries", Json::array()}};
        for (const auto &e : model.Elements) points["elements"].push_back({{"tag", e.Tag}, {"points", Coordinates(e.Points)}});
        for (const auto &f : model.Faces)
            if (f.Right < 0) points["boundaries"].push_back({{"tag", model.Elements[f.Left].Tag}, {"points", Coordinates(f.Points)}});
        std::ofstream out(argv[2]);
        out << points.dump() << '\n';
        if (!out) throw std::runtime_error("Cannot write points");
        return 0;
    }
    const Json data = Read(argv[4]);
    const double omega = data.at("omega");
    if (!std::isfinite(omega) || omega <= 0) throw std::runtime_error("Invalid frequency");
    std::array<Vec, 2> reference, forcing;
    for (int part = 0; part < 2; ++part) {
        const char *key = part ? "imag" : "real";
        int element = 0, row = 0;
        reference[part] = model.Project([&](const V3 &, int) -> Vec {
            const auto &e = model.Elements.at(element);
            auto value = Row(data.at("elements").at(element).at(key), row++, e.Fields());
            if (row == e.Points.rows()) {
                ++element;
                row = 0;
            }
            return value;
        });
        std::map<std::array<double, 4>, Vec> boundary;
        int index = 0;
        for (const auto &f : model.Faces)
            if (f.Right < 0) {
                const auto &e = model.Elements[f.Left];
                for (int q = 0; q < f.Points.rows(); ++q) boundary[{f.Points(q, 0), f.Points(q, 1), f.Points(q, 2), double(e.Tag)}] = Row(data.at("boundaries").at(index).at(key), q, e.Fields());
                ++index;
            }
        model.BoundaryState = [&](const V3 &x, int tag, double) -> Vec { return boundary.at({x[0], x[1], x[2], double(tag)}); };
        forcing[part] = model.Rhs(Vec::Zero(model.Size));
    }
    model.BoundaryState = {};
    Eigen::VectorXcd state;
    double residual = 0;
    const bool trajectory = config.contains("reference_seconds");
    if (trajectory) {
        const double duration = config.at("reference_seconds"), rate = config.value("sample_rate", 1000.);
        model.Initial = reference[0];
        model.PrescribedLoad = [&](double t) -> Vec { return std::cos(omega * t) * forcing[0] + std::sin(omega * t) * forcing[1]; };
        const auto trace = model.RunReference(duration, rate, config.value("dt", 0.));
        state = trace.State.cast<Complex>();
        const double endTime = trace.Samples.rows() / rate;
        reference[0] = std::cos(omega * endTime) * reference[0] + std::sin(omega * endTime) * reference[1];
        const dg::reference::Mat samples = trace.Samples;
        std::ofstream out(std::string(argv[3]) + ".bin", std::ios::binary);
        out.write(reinterpret_cast<const char *>(samples.data()), samples.size() * sizeof(double));
        if (!out) throw std::runtime_error("Cannot write reference trace");
    } else {
        Eigen::SparseMatrix<Complex> a = -model.System.cast<Complex>();
        for (int i = 0; i < model.Size; ++i) a.coeffRef(i, i) -= Complex(0, omega);
        a.makeCompressed();
        Eigen::SparseLU<Eigen::SparseMatrix<Complex>> solver;
        solver.compute(a);
        if (solver.info() != Eigen::Success) throw std::runtime_error("Reference system factorization failed");
        const Eigen::VectorXcd rhs = forcing[0].cast<Complex>() + Complex(0, 1) * forcing[1].cast<Complex>();
        state = solver.solve(rhs);
        if (solver.info() != Eigen::Success || !state.allFinite()) throw std::runtime_error("Reference solve failed");
        residual = (a * state - rhs).norm() / rhs.norm();
    }
    std::map<int, std::array<double, 2>> errors;
    for (const auto &e : model.Elements)
        for (int part = 0; part < (trajectory ? 1 : 2); ++part) {
            const Vec r = reference[part].segment(e.Offset, model.Nodes * e.Fields());
            const Vec value = part ? Vec(state.segment(e.Offset, r.size()).imag()) : Vec(state.segment(e.Offset, r.size()).real());
            const Vec delta = value - r;
            errors[e.Tag][0] += delta.dot(e.PhysicalMass * delta);
            errors[e.Tag][1] += r.dot(e.PhysicalMass * r);
        }
    Json report = {{"linear_residual", residual}, {"unknowns", model.Size}, {"degree", model.Degree}, {"omega", omega}};
    for (const auto &[tag, error] : errors) report["relative_errors"][std::to_string(tag)] = std::sqrt(error[0] / error[1]);
    std::cout << report.dump(2) << '\n';
    std::ofstream out(argv[3]);
    out << report.dump(2) << '\n';
    if (!out || residual > 1e-9) throw std::runtime_error("Reference comparison failed");
    return 0;
} catch (const std::exception &e) {
    std::cerr << "ElasticReference: " << e.what() << '\n';
    return 1;
}
