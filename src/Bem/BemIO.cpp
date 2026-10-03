#include "BemIO.h"
#include <limits>

namespace bem {
using json = nlohmann::json;
Problem ReadProblem(const json &j) {
    Problem p;
    const auto readPoints = [](const json &a) {
        Points points(a.size(), 3);
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].size() != 3) throw std::invalid_argument("BEM point needs three coordinates");
            for (int k = 0; k < 3; ++k) points(i, k) = a[i][k].get<double>();
        }
        return points;
    };
    p.Vertices = readPoints(j.at("vertices"));
    p.Listeners = readPoints(j.at("listeners"));
    const auto index = [&](const json &value) {
        if (!value.is_number_integer()) throw std::invalid_argument("BEM indices must be integers");
        const int64_t i = value.get<int64_t>();
        if (i < 0 || i >= p.Vertices.rows() || i > std::numeric_limits<int>::max()) throw std::invalid_argument("BEM vertex index out of bounds");
        return int(i);
    };
    const auto &faces = j.at("triangles");
    p.Faces.resize(faces.size(), 3);
    for (size_t i = 0; i < faces.size(); ++i) {
        if (faces[i].size() != 3) throw std::invalid_argument("BEM face needs three indices");
        for (int k = 0; k < 3; ++k) p.Faces(i, k) = index(faces[i][k]);
    }
    if (j.at("source").size() != 3) throw std::invalid_argument("BEM source needs three coordinates");
    for (int k = 0; k < 3; ++k) p.Source[k] = j.at("source")[k].get<double>();
    p.Frequency = j.at("frequency_hz").get<double>();
    p.SoundSpeed = j.value("sound_speed", 344.);
    if (j.contains("height_vertices"))
        for (const auto &v : j.at("height_vertices")) p.HeightVertices.push_back(index(v));
    Validate(p);
    return p;
}
json WriteResult(const Result &r, bool includeFields) {
    json output = {{"diffusion", r.Diffusion}, {"forward_residual", r.ForwardResidual}, {"adjoint_residual", r.AdjointResidual}, {"assembly_seconds", r.AssemblySeconds}, {"solve_seconds", r.SolveSeconds}, {"derivative_seconds", r.DerivativeSeconds}};
    if (includeFields) {
        const auto complex = [](const Vector &v) {
            json a = json::array();
            for (const auto x : v) a.push_back({x.real(), x.imag()});
            return a;
        };
        output["surface_pressure"] = complex(r.SurfacePressure);
        output["scattered_pressure"] = complex(r.ScatteredPressure);
        std::vector<double> gradient(r.HeightGradient.size());
        for (size_t i = 0; i < gradient.size(); ++i) gradient[i] = r.HeightGradient[i];
        output["height_gradient"] = gradient;
    }
    const auto stats = [](const CompressionStats &s) {
        return json{{"dense_entries", s.DenseEntries}, {"stored_entries", s.StoredEntries}, {"blocks", s.Blocks}, {"low_rank_blocks", s.LowRankBlocks}, {"max_rank", s.MaxRank}, {"relative_frobenius_error", s.RelativeError}, {"max_block_error", s.MaxBlockError}};
    };
    output["boundary_compression"] = stats(r.BoundaryCompression);
    output["listener_compression"] = stats(r.ListenerCompression);
    const auto &assembly = r.Assembly;
    output["assembly_profile"] = {{"quadrature_seconds", assembly.QuadratureSeconds}, {"compression_seconds", assembly.CompressionSeconds}, {"packing_seconds", assembly.PackingSeconds}, {"preconditioner_seconds", assembly.PreconditionerSeconds}, {"other_seconds", assembly.OtherSeconds}};
    output["forward_iterations"] = r.ForwardIterations;
    const auto &pre = r.Preconditioning;
    output["preconditioning"] = {{"method", pre.Kind == Preconditioner::None ? "none" : pre.Kind == Preconditioner::BlockJacobi ? "block_jacobi" :
                                                                                                                                  "two_level"},
                                 {"setup_seconds", pre.Seconds},
                                 {"stored_entries", pre.StoredEntries},
                                 {"local_blocks", pre.LocalBlocks},
                                 {"coarse_size", pre.CoarseSize}};
    output["adjoint_iterations"] = r.AdjointIterations;
    output["independent_sampled_rows"] = r.IndependentRows;
    output["independent_forward_residual"] = r.IndependentForwardResidual;
    output["independent_adjoint_residual"] = r.IndependentAdjointResidual;
    return output;
}
} // namespace bem
