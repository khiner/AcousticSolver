// Certified ACA follows the corrected Acoustic Reliefs reference. See LICENSE.AcousticReliefs.
#include "Bem.h"
#include "BemGpu.h"
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <Eigen/LU>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <limits>
#include <numeric>
#include <thread>

namespace bem::detail {
// Dense blocks are retained when a certified factorization would cost more storage.
struct CompressedBlock {
    Matrix U, V;
    double Error{};
    bool Dense{};
};
struct IterativeResult {
    Vector Value;
    double Residual{};
    int Iterations{};
};
namespace {
struct SingleThreadBlas {
    BLAS_THREADING Previous{BLASGetThreading()};
    SingleThreadBlas() { BLASSetThreading(BLAS_THREADING_SINGLE_THREADED); }
    ~SingleThreadBlas() { BLASSetThreading(Previous); }
};
} // namespace
static CompressedBlock Compress(const Matrix &a, double tolerance) {
    if (!(tolerance > 0.) || !std::isfinite(tolerance) || !a.allFinite()) throw std::invalid_argument("Invalid BEM compression input");
    const int m = a.rows(), n = a.cols();
    const double norm = a.norm();
    if (norm == 0.) return {Matrix(m, 0), Matrix(0, n), 0., false};
    // Compression already runs across independent blocks on up to eight workers.
    const SingleThreadBlas threading;
    const int limit = int(int64_t(m) * n / (m + n));
    Matrix u(m, limit), v(limit, n);
    int rank = 0, pivotRow = 0;
    double update = 0.;
    std::vector<bool> used(m, false);
    while (rank < limit) {
        Eigen::RowVectorXcd row = a.row(pivotRow) - u.row(pivotRow).head(rank) * v.topRows(rank);
        Eigen::Index col;
        const double pivot = row.cwiseAbs2().maxCoeff(&col);
        if (pivot > 0.) {
            row /= row[col];
            const Vector column = a.col(col) - u.leftCols(rank) * v.col(col).head(rank);
            u.col(rank) = column;
            v.row(rank) = row;
            update = column.norm() * row.norm();
            used[pivotRow] = true;
            ++rank;
            double largest = -1.;
            for (int i = 0; i < m; ++i)
                if (!used[i] && std::norm(column[i]) > largest) {
                    largest = std::norm(column[i]);
                    pivotRow = i;
                }
        }
        if (pivot == 0. || update <= tolerance * norm || rank == limit) {
            // Certify the actual FP32 factors uploaded to Metal, over the entire block.
            // The largest residual row restarts ACA across disconnected interactions.
            const Matrix packedU = u.leftCols(rank).cast<std::complex<float>>().cast<std::complex<double>>();
            const Matrix packedV = v.topRows(rank).cast<std::complex<float>>().cast<std::complex<double>>();
            Matrix residual = a;
            const std::complex<double> minusOne{-1., 0.}, one{1., 0.};
            if (rank) cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, rank, &minusOne, packedU.data(), m, packedV.data(), rank, &one, residual.data(), m);
            const double error = residual.norm() / norm;
            if (error <= tolerance) return {packedU, packedV, error, false};
            Eigen::Index next;
            residual.rowwise().squaredNorm().maxCoeff(&next);
            pivotRow = int(next);
            // Retain dense storage when the largest packed residual lies on an exhausted ACA row.
            if (rank > 0 && (a.row(pivotRow) - u.row(pivotRow).head(rank) * v.topRows(rank)).norm() <= std::numeric_limits<double>::epsilon() * norm) break;
        }
    }
    return {a, Matrix{}, 0., true};
}

namespace {
constexpr uint32_t DenseRank = std::numeric_limits<uint32_t>::max();
struct Node {
    int Begin{}, Count{}, Left{-1}, Right{-1};
    Eigen::Vector3d Min, Max, Center;
    double Radius{};
};
struct Tree {
    std::vector<int> Order;
    std::vector<Node> Nodes;
    Points Centers, Lower, Upper;
    Tree(const Problem &p, bool faces, int leaf) {
        const int count = faces ? p.Faces.rows() : p.Listeners.rows();
        Order.resize(count);
        std::iota(Order.begin(), Order.end(), 0);
        Centers.resize(count, 3);
        Lower.resize(count, 3);
        Upper.resize(count, 3);
        for (int i = 0; i < count; ++i) {
            if (faces) {
                const Eigen::Vector3d a = p.Vertices.row(p.Faces(i, 0)), b = p.Vertices.row(p.Faces(i, 1)), c = p.Vertices.row(p.Faces(i, 2));
                Centers.row(i) = (a + b + c) / 3.;
                Lower.row(i) = a.cwiseMin(b).cwiseMin(c);
                Upper.row(i) = a.cwiseMax(b).cwiseMax(c);
            } else Centers.row(i) = Lower.row(i) = Upper.row(i) = p.Listeners.row(i);
        }
        Build(0, count, leaf);
    }
    int Build(int begin, int count, int leaf) {
        Node node;
        node.Begin = begin;
        node.Count = count;
        node.Min = Lower.row(Order[begin]);
        node.Max = Upper.row(Order[begin]);
        for (int i = begin + 1; i < begin + count; ++i) {
            node.Min = node.Min.cwiseMin(Lower.row(Order[i]).transpose());
            node.Max = node.Max.cwiseMax(Upper.row(Order[i]).transpose());
        }
        node.Center.setZero();
        for (int i = begin; i < begin + count; ++i) node.Center += Centers.row(Order[i]).transpose();
        node.Center /= count;
        for (int i = begin; i < begin + count; ++i) node.Radius = std::max(node.Radius, (Centers.row(Order[i]).transpose() - node.Center).norm());
        const int index = Nodes.size();
        Nodes.push_back(node);
        if (count > leaf) {
            Eigen::Index axis;
            (node.Max - node.Min).maxCoeff(&axis);
            const auto less = [&](int a, int b) { return Centers(a, axis) != Centers(b, axis) ? Centers(a, axis) < Centers(b, axis) : a < b; };
            std::nth_element(Order.begin() + begin, Order.begin() + begin + count / 2, Order.begin() + begin + count, less);
            const int left = Build(begin, count / 2, leaf), right = Build(begin + count / 2, count - count / 2, leaf);
            Nodes[index].Left = left;
            Nodes[index].Right = right;
        }
        return index;
    }
};
struct Region {
    int Row, Col, Rows, Cols;
    bool Approximate;
};
void Partition(const Tree &rows, const Tree &cols, int r, int c, int maximum, std::vector<Region> &blocks) {
    const auto &a = rows.Nodes[r], &b = cols.Nodes[c];
    // The reference's 1.5-radius separation selects candidates; full residual
    // certification, rather than this geometric heuristic, accepts their factors.
    const bool far = (a.Center - b.Center).norm() > 1.5 * (a.Radius + b.Radius);
    if ((far && std::max(a.Count, b.Count) <= maximum) || (a.Left < 0 && b.Left < 0)) {
        blocks.push_back({a.Begin, b.Begin, a.Count, b.Count, far});
    } else if (a.Left >= 0 && (b.Left < 0 || a.Count >= b.Count)) {
        Partition(rows, cols, a.Left, c, maximum, blocks);
        Partition(rows, cols, a.Right, c, maximum, blocks);
    } else {
        Partition(rows, cols, r, b.Left, maximum, blocks);
        Partition(rows, cols, r, b.Right, maximum, blocks);
    }
}
template<class T> void Upload(GpuBuffer &buffer, const std::vector<T> &data) {
    buffer.Resize(std::max(size_t(1), data.size()) * sizeof(T));
    if (!data.empty()) buffer.Upload(data.data(), data.size() * sizeof(T));
}
void Append(std::vector<Complex> &data, const Matrix &a) {
    for (int i = 0; i < a.size(); ++i) data.push_back({float(a.data()[i].real()), float(a.data()[i].imag())});
}
void GatherLayout(const std::vector<BemBlock> &blocks, int size, bool transpose, GpuBuffer &offsets, GpuBuffer &indices) {
    std::vector<std::vector<uint32_t>> lists(size);
    for (const auto &b : blocks)
        for (uint32_t j = 0; j < (transpose ? b.Cols : b.Rows); ++j)
            lists[(transpose ? b.Col : b.Row) + j].push_back(b.Output + j);
    std::vector<uint32_t> starts{0}, entries;
    for (const auto &list : lists) {
        entries.insert(entries.end(), list.begin(), list.end());
        starts.push_back(entries.size());
    }
    Upload(offsets, starts);
    Upload(indices, entries);
}
void CheckOptions(const SolverOptions &o) {
    if (o.Precondition != Preconditioner::None && o.Precondition != Preconditioner::BlockJacobi && o.Precondition != Preconditioner::TwoLevel) throw std::invalid_argument("Unknown BEM preconditioner");
    if (!(o.CompressionTolerance > 0. && o.CompressionTolerance < 1.) || !std::isfinite(o.CompressionTolerance) || !(o.SolveTolerance > 0. && o.SolveTolerance < 1.) || !std::isfinite(o.SolveTolerance) || o.LeafSize < 4 || o.MaxBlockSize < o.LeafSize || o.MaxBlockSize > 4096 || o.Restart < 1 || o.Restart > 200 || o.MaxIterations < 1) throw std::invalid_argument("Invalid BEM iterative/compression settings");
}
} // namespace

struct MetalOperator {
    Tree Rows, Cols;
    GpuBuffer Blocks, Values, Temporary, Partial, Input, Output;
    GpuBuffer RowOffsets, RowIndices, ColOffsets, ColIndices;
    BemBlockParams Params{};
    CompressionStats Statistics;
    AssemblyProfile Timings;
    GpuBuffer InverseBlocks, InverseValues;
    uint32_t InverseCount{};
    double PreconditionerTime{};
    Preconditioner Kind{Preconditioner::None};
    uint64_t InverseEntries{};
    GpuBuffer CoarseGroups, FineGroups, CoarseInverse, CoarseCorrection, CoarseInput, CoarseOutput;
    uint32_t CoarseCount{};
    MetalOperator(const Problem &p, bool boundary, const SolverOptions &options) : Rows(p, boundary, options.LeafSize), Cols(p, true, options.LeafSize) {
        const Geometry geometry(p);
        GpuBuffer rowOrder, colOrder, assembled;
        Upload(rowOrder, Rows.Order);
        Upload(colOrder, Cols.Order);
        std::vector<Region> regions;
        Partition(Rows, Cols, 0, 0, options.MaxBlockSize, regions);
        auto &ctx = MetalContext::Get();
        auto *assembly = ctx.BemPipeline("bem_assemble_block");
        std::vector<BemBlock> blocks;
        std::vector<Complex> values;
        std::vector<BemBlock> inverseBlocks;
        std::vector<Complex> inverseValues;
        std::vector<BemCoarseGroup> groups;
        std::vector<uint32_t> groupIndex(Rows.Order.size());
        std::vector<Matrix> localInverses;
        Matrix ap;
        const auto preStart = std::chrono::steady_clock::now();
        if (boundary && options.Precondition == Preconditioner::TwoLevel) {
            for (const auto &node : Rows.Nodes)
                if (node.Left < 0) groups.push_back({uint32_t(node.Begin), uint32_t(node.Count), float(1. / std::sqrt(node.Count)), 0.f});
            if (groups.size() > 1024) throw std::invalid_argument("BEM coarse preconditioner exceeds 1024 groups; increase leaf size");
            for (size_t c = 0; c < groups.size(); ++c)
                for (uint32_t i = groups[c].Begin; i < groups[c].Begin + groups[c].Count; ++i) groupIndex[i] = c;
            ap = Matrix::Zero(Rows.Order.size(), groups.size());
            localInverses.resize(groups.size());
        }
        PreconditionerTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - preStart).count();
        uint64_t temporaryCount = 0, outputCount = 0;
        double errorSquared = 0., normSquared = 0.;
        // Bound scratch storage and amortize GPU synchronization over many blocks.
        // Workers only read completed GPU storage and write distinct result slots.
        for (size_t begin = 0; begin < regions.size();) {
            const auto quadratureStart = std::chrono::steady_clock::now();
            std::vector<size_t> offsets{0};
            size_t end = begin;
            do {
                offsets.push_back(offsets.back() + size_t(regions[end].Rows) * regions[end].Cols);
                ++end;
            } while (end < regions.size() && end - begin < 256 && offsets.back() + size_t(regions[end].Rows) * regions[end].Cols <= 4194304);
            assembled.Resize(offsets.back() * sizeof(Complex));
            for (size_t j = begin; j < end; ++j) {
                const auto &region = regions[j];
                const BemBlockParams params{uint32_t(region.Row), uint32_t(region.Col), uint32_t(region.Rows), uint32_t(region.Cols), uint32_t(boundary), 0, 0, 0, geometry.Params.Wavenumber};
                const uint32_t count = params.Rows * params.Cols;
                ctx.Dispatch(assembly, {(count + 127) / 128, 1, 1}, {128, 1, 1}, {&geometry.Faces, &geometry.Listeners, &rowOrder, &colOrder, GpuSlice(&assembled, offsets[j - begin] * sizeof(Complex)), &geometry.ListenerPhases}, &params, sizeof(params));
            }
            const auto *data = assembled.As<Complex>();
            Timings.QuadratureSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - quadratureStart).count();
            const auto compressionStart = std::chrono::steady_clock::now();
            std::vector<CompressedBlock> packed(end - begin);
            std::vector<double> norms(end - begin);
            std::atomic<size_t> next{0};
            const auto compress = [&] {
                for (size_t j = next.fetch_add(1); j < packed.size(); j = next.fetch_add(1)) {
                    const auto &region = regions[begin + j];
                    Matrix a(region.Rows, region.Cols);
                    for (int i = 0; i < a.size(); ++i) a.data()[i] = {data[offsets[j] + i].Real, data[offsets[j] + i].Imag};
                    if (!a.allFinite()) throw std::runtime_error("Nonfinite BEM block");
                    norms[j] = a.squaredNorm();
                    packed[j] = region.Approximate ? Compress(a, options.CompressionTolerance) : CompressedBlock{std::move(a), Matrix{}, 0., true};
                }
            };
            if (offsets.back() < 65536) compress();
            else {
                std::vector<std::future<void>> workers;
                const size_t count = std::min({size_t(8), size_t(std::max(1u, std::thread::hardware_concurrency())), packed.size()});
                workers.reserve(count);
                for (size_t i = 0; i < count; ++i) workers.push_back(std::async(std::launch::async, compress));
                for (auto &worker : workers) worker.get();
            }
            Timings.CompressionSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - compressionStart).count();
            const auto packingStart = std::chrono::steady_clock::now();
            const double preBeforePacking = PreconditionerTime;
            for (size_t j = 0; j < packed.size(); ++j) {
                const auto &region = regions[begin + j];
                const auto &compressed = packed[j];
                if (boundary && options.Precondition != Preconditioner::None && region.Row == region.Col) {
                    const auto start = std::chrono::steady_clock::now();
                    if (!compressed.Dense || region.Rows != region.Cols) throw std::runtime_error("Invalid BEM diagonal block");
                    const Matrix inverse = compressed.U.partialPivLu().inverse();
                    if (!inverse.cast<std::complex<float>>().allFinite() || (compressed.U * inverse - Matrix::Identity(region.Rows, region.Rows)).norm() > 1e-9 * std::sqrt(region.Rows)) throw std::runtime_error("Singular BEM preconditioner block");
                    inverseBlocks.push_back({uint32_t(region.Row), uint32_t(region.Col), uint32_t(region.Rows), uint32_t(region.Cols), uint32_t(inverseValues.size()), DenseRank, 0, uint32_t(region.Row)});
                    Append(inverseValues, inverse);
                    if (!groups.empty()) localInverses[groupIndex[region.Row]] = inverse;
                    PreconditionerTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                }
                if (!groups.empty()) {
                    const auto start = std::chrono::steady_clock::now();
                    // P has a normalized constant on each spatial leaf. Accumulate
                    // A*P from certified factors while they are already on the host.
                    for (int col = region.Col; col < region.Col + region.Cols;) {
                        const uint32_t c = groupIndex[col];
                        const int count = std::min(region.Col + region.Cols, int(groups[c].Begin + groups[c].Count)) - col;
                        Vector sum;
                        if (compressed.Dense) sum = compressed.U.middleCols(col - region.Col, count).rowwise().sum();
                        else sum = compressed.U * compressed.V.middleCols(col - region.Col, count).rowwise().sum();
                        ap.block(region.Row, c, region.Rows, 1) += sum / std::sqrt(groups[c].Count);
                        col += count;
                    }
                    PreconditionerTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                }
                const uint32_t rank = compressed.Dense ? DenseRank : uint32_t(compressed.U.cols());
                const uint64_t entries = compressed.U.size() + compressed.V.size();
                if (values.size() + entries >= DenseRank || temporaryCount + (compressed.Dense ? 0 : rank) >= DenseRank || outputCount + std::max(region.Rows, region.Cols) >= DenseRank) throw std::runtime_error("BEM compressed storage exceeds 32-bit indexing");
                blocks.push_back({uint32_t(region.Row), uint32_t(region.Col), uint32_t(region.Rows), uint32_t(region.Cols), uint32_t(values.size()), rank, uint32_t(temporaryCount), uint32_t(outputCount)});
                Append(values, compressed.U);
                Append(values, compressed.V);
                temporaryCount += compressed.Dense ? 0 : rank;
                outputCount += std::max(region.Rows, region.Cols);
                Statistics.LowRankBlocks += !compressed.Dense;
                if (!compressed.Dense) Statistics.MaxRank = std::max(Statistics.MaxRank, int(rank));
                Statistics.MaxBlockError = std::max(Statistics.MaxBlockError, compressed.Error);
                normSquared += norms[j];
                errorSquared += compressed.Error * compressed.Error * norms[j];
            }
            Timings.PackingSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - packingStart).count() - (PreconditionerTime - preBeforePacking);
            begin = end;
        }
        const auto uploadStart = std::chrono::steady_clock::now();
        Statistics.DenseEntries = uint64_t(Rows.Order.size()) * Cols.Order.size();
        Statistics.StoredEntries = values.size();
        Statistics.Blocks = blocks.size();
        Statistics.RelativeError = std::sqrt(errorSquared / std::max(normSquared, 1e-300));
        Upload(Blocks, blocks);
        Upload(Values, values);
        Temporary.Resize(std::max(uint64_t(1), temporaryCount) * sizeof(BemFloat4));
        Partial.Resize(outputCount * sizeof(BemFloat4));
        const auto size = std::max(Rows.Order.size(), Cols.Order.size());
        Input.Resize(size * sizeof(BemFloat4));
        Output.Resize(size * sizeof(BemFloat4));
        GatherLayout(blocks, Rows.Order.size(), false, RowOffsets, RowIndices);
        GatherLayout(blocks, Cols.Order.size(), true, ColOffsets, ColIndices);
        Params.BlockCount = blocks.size();
        InverseCount = inverseBlocks.size();
        Kind = boundary ? options.Precondition : Preconditioner::None;
        InverseEntries = inverseValues.size();
        Timings.PackingSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - uploadStart).count();
        const auto inverseUploadStart = std::chrono::steady_clock::now();
        Upload(InverseBlocks, inverseBlocks);
        Upload(InverseValues, inverseValues);
        PreconditionerTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - inverseUploadStart).count();
        if (!groups.empty()) {
            const auto start = std::chrono::steady_clock::now();
            CoarseCount = groups.size();
            Matrix coarse(CoarseCount, CoarseCount), correction(ap.rows(), ap.cols());
            for (uint32_t c = 0; c < CoarseCount; ++c) {
                const auto &g = groups[c];
                coarse.row(c) = ap.middleRows(g.Begin, g.Count).colwise().sum() / std::sqrt(g.Count);
                correction.middleRows(g.Begin, g.Count) = -localInverses[c] * ap.middleRows(g.Begin, g.Count);
                correction.block(g.Begin, c, g.Count, 1).array() += 1. / std::sqrt(g.Count);
            }
            const Matrix inverse = coarse.partialPivLu().inverse();
            if (!inverse.cast<std::complex<float>>().allFinite() || !correction.cast<std::complex<float>>().allFinite() || (coarse * inverse - Matrix::Identity(CoarseCount, CoarseCount)).norm() > 1e-8 * std::sqrt(CoarseCount)) throw std::runtime_error("Singular BEM coarse preconditioner");
            std::vector<Complex> packedInverse, packedCorrection;
            Append(packedInverse, inverse);
            Append(packedCorrection, correction);
            Upload(CoarseGroups, groups);
            Upload(FineGroups, groupIndex);
            Upload(CoarseInverse, packedInverse);
            Upload(CoarseCorrection, packedCorrection);
            CoarseInput.Resize(CoarseCount * sizeof(BemFloat4));
            CoarseOutput.Resize(CoarseCount * sizeof(BemFloat4));
            PreconditionerTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        Timings.PreconditionerSeconds = PreconditionerTime;
    }
    Vector Apply(const Vector &x, bool transpose = false) {
        const auto &inputOrder = transpose ? Rows.Order : Cols.Order;
        const auto &outputOrder = transpose ? Cols.Order : Rows.Order;
        if (x.size() != int(inputOrder.size()) || !x.allFinite()) throw std::invalid_argument("Invalid BEM operator vector");
        auto *packed = static_cast<BemFloat4 *>(Input.RawData());
        for (size_t i = 0; i < inputOrder.size(); ++i) {
            const auto value = x[inputOrder[i]];
            const float real = float(value.real()), imag = float(value.imag());
            packed[i] = {real, imag, float(value.real() - double(real)), float(value.imag() - double(imag))};
        }
        Params.Transpose = transpose;
        Params.OutputCount = outputOrder.size();
        auto &ctx = MetalContext::Get();
        ctx.Dispatch(ctx.BemPipeline("bem_block_reduce"), {Params.BlockCount, 1, 1}, {128, 1, 1}, {&Blocks, &Values, &Input, &Temporary}, &Params, sizeof(Params));
        ctx.Dispatch(ctx.BemPipeline("bem_block_product"), {Params.BlockCount, 1, 1}, {128, 1, 1}, {&Blocks, &Values, &Input, &Temporary, &Partial}, &Params, sizeof(Params));
        ctx.Dispatch(ctx.BemPipeline("bem_block_gather"), {(Params.OutputCount + 127) / 128, 1, 1}, {128, 1, 1}, {transpose ? &ColOffsets : &RowOffsets, transpose ? &ColIndices : &RowIndices, &Partial, &Output}, &Params, sizeof(Params));
        const auto *data = Output.As<BemFloat4>();
        Vector out(outputOrder.size());
        for (size_t i = 0; i < outputOrder.size(); ++i) out[outputOrder[i]] = {double(data[i].x) + data[i].z, double(data[i].y) + data[i].w};
        if (!out.allFinite()) throw std::runtime_error("Nonfinite BEM operator product");
        return out;
    }
    Vector Precondition(const Vector &x, bool transpose = false) {
        if (x.size() != int(Rows.Order.size()) || !x.allFinite()) throw std::invalid_argument("Invalid BEM preconditioner vector");
        if (!InverseCount) throw std::invalid_argument("BEM boundary preconditioner unavailable");
        auto *packed = Input.As<BemFloat4>();
        for (size_t i = 0; i < Rows.Order.size(); ++i) {
            const auto v = x[Rows.Order[i]];
            const float real = float(v.real()), imag = float(v.imag());
            packed[i] = {real, imag, float(v.real() - real), float(v.imag() - imag)};
        }
        auto params = Params;
        params.Transpose = transpose;
        params.BlockCount = InverseCount;
        auto &ctx = MetalContext::Get();
        ctx.Dispatch(ctx.BemPipeline("bem_block_product"), {InverseCount, 1, 1}, {128, 1, 1}, {&InverseBlocks, &InverseValues, &Input, &Temporary, &Output}, &params, sizeof(params));
        if (CoarseCount) {
            const BemCoarseParams coarse{uint32_t(x.size()), CoarseCount, uint32_t(transpose), 0};
            ctx.Dispatch(ctx.BemPipeline("bem_coarse_restrict"), {CoarseCount, 1, 1}, {128, 1, 1}, {&CoarseGroups, &CoarseCorrection, &Input, &CoarseInput}, &coarse, sizeof(coarse));
            ctx.Dispatch(ctx.BemPipeline("bem_coarse_solve"), {(CoarseCount + 63) / 64, 1, 1}, {64, 1, 1}, {&CoarseInverse, &CoarseInput, &CoarseOutput}, &coarse, sizeof(coarse));
            ctx.Dispatch(ctx.BemPipeline("bem_coarse_add"), {(uint32_t(x.size()) + 63) / 64, 1, 1}, {64, 1, 1}, {&CoarseGroups, &FineGroups, &CoarseCorrection, &CoarseOutput, &Output}, &coarse, sizeof(coarse));
        }
        const auto *data = Output.As<BemFloat4>();
        Vector result(x.size());
        for (size_t i = 0; i < Rows.Order.size(); ++i) result[Rows.Order[i]] = {double(data[i].x) + data[i].z, double(data[i].y) + data[i].w};
        if (!result.allFinite()) throw std::runtime_error("Nonfinite BEM preconditioner result");
        return result;
    }
    PreconditionerStats Preconditioning() const { return {Kind, PreconditionerTime, InverseEntries + uint64_t(CoarseCount) * (Rows.Order.size() + CoarseCount), int(InverseCount), int(CoarseCount)}; }
    const CompressionStats &Stats() const { return Statistics; }
    const AssemblyProfile &Profile() const { return Timings; }
};

static IterativeResult Gmres(const std::function<Vector(const Vector &)> &apply, const Vector &rhs, const SolverOptions &options, const std::function<Vector(const Vector &)> &precondition) {
    // Right preconditioning changes the search directions. Every stopping check
    // still evaluates the residual of the original, unpreconditioned equations.
    if (!rhs.allFinite()) throw std::invalid_argument("Nonfinite BEM right-hand side");
    IterativeResult result{Vector::Zero(rhs.size()), 0., 0};
    const double norm = rhs.norm();
    if (norm == 0.) return result;
    const int restart = std::min(int(rhs.size()), options.Restart);
    Matrix basis(rhs.size(), restart + 1), h(restart + 1, restart);
    Vector residual = rhs;
    std::vector<double> cosine(restart);
    std::vector<std::complex<double>> sine(restart);
    while (result.Iterations < options.MaxIterations) {
        const double beta = residual.norm();
        basis.col(0) = residual / beta;
        h.setZero();
        Vector target = Vector::Zero(restart + 1);
        target[0] = beta;
        const Vector initial = result.Value;
        for (int j = 0; j < restart && result.Iterations < options.MaxIterations; ++j) {
            Vector w = apply(precondition ? precondition(basis.col(j)) : Vector(basis.col(j)));
            // Reorthogonalize in FP64; operator products and storage remain on Metal.
            for (int pass = 0; pass < 2; ++pass)
                for (int i = 0; i <= j; ++i) {
                    const auto coefficient = basis.col(i).dot(w);
                    h(i, j) += coefficient;
                    w -= coefficient * basis.col(i);
                }
            const double next = w.norm();
            h(j + 1, j) = next;
            if (next > 0.) basis.col(j + 1) = w / next;
            // Incremental complex Givens rotations avoid refactoring the growing
            // Hessenberg system at every iteration, especially at high frequencies.
            for (int i = 0; i < j; ++i) {
                const auto upper = cosine[i] * h(i, j) + sine[i] * h(i + 1, j);
                h(i + 1, j) = -std::conj(sine[i]) * h(i, j) + cosine[i] * h(i + 1, j);
                h(i, j) = upper;
            }
            const double diagonal = std::abs(h(j, j)), lower = std::abs(h(j + 1, j));
            const double length = std::hypot(diagonal, lower);
            if (length == 0.) throw std::runtime_error("BEM GMRES Arnoldi breakdown before convergence");
            const std::complex<double> phase = diagonal > 0. ? h(j, j) / diagonal : std::complex<double>{1., 0.};
            cosine[j] = diagonal / length;
            sine[j] = phase * std::conj(h(j + 1, j)) / length;
            h(j, j) = phase * length;
            h(j + 1, j) = 0.;
            target[j + 1] = -std::conj(sine[j]) * target[j];
            target[j] *= cosine[j];
            const double estimated = std::abs(target[j + 1]) / norm;
            ++result.Iterations;
            if (estimated <= .25 * options.SolveTolerance || next < 1e-14 || j + 1 == restart || result.Iterations == options.MaxIterations) {
                const Vector coefficients = h.topLeftCorner(j + 1, j + 1).triangularView<Eigen::Upper>().solve(target.head(j + 1));
                const Vector correction = basis.leftCols(j + 1) * coefficients;
                result.Value = initial + (precondition ? precondition(correction) : correction);
                residual = rhs - apply(result.Value);
                result.Residual = residual.norm() / norm;
                if (result.Value.allFinite() && result.Residual <= options.SolveTolerance) return result;
                break;
            }
        }
    }
    throw std::runtime_error("BEM GMRES failed actual residual check after " + std::to_string(result.Iterations) + " iterations: " + std::to_string(result.Residual));
}
} // namespace bem::detail

namespace bem {
Result SolveMetal(const Problem &p, const SolverOptions &options) {
    using Clock = std::chrono::steady_clock;
    const auto seconds = [](Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); };
    const auto start = Clock::now();
    Validate(p);
    detail::CheckOptions(options);
    detail::MetalOperator boundary(p, true, options), listener(p, false, options);
    Result result;
    result.AssemblySeconds = seconds(start);
    for (const auto &profile : {boundary.Profile(), listener.Profile()}) {
        result.Assembly.QuadratureSeconds += profile.QuadratureSeconds;
        result.Assembly.CompressionSeconds += profile.CompressionSeconds;
        result.Assembly.PackingSeconds += profile.PackingSeconds;
        result.Assembly.PreconditionerSeconds += profile.PreconditionerSeconds;
    }
    result.Assembly.OtherSeconds = result.AssemblySeconds - result.Assembly.QuadratureSeconds - result.Assembly.CompressionSeconds - result.Assembly.PackingSeconds - result.Assembly.PreconditionerSeconds;
    result.Preconditioning = boundary.Preconditioning();
    result.BoundaryCompression = boundary.Stats();
    result.ListenerCompression = listener.Stats();
    const auto solveStart = Clock::now();
    const Vector incident = detail::SourceValues(p, false);
    const std::function<Vector(const Vector &)> precondition = options.Precondition == Preconditioner::None ? std::function<Vector(const Vector &)>{} : [&](const Vector &v) { return boundary.Precondition(v); };
    const auto forward = detail::Gmres([&](const Vector &v) { return boundary.Apply(v); }, -incident, options, precondition);
    result.SurfacePressure = forward.Value;
    result.ForwardResidual = forward.Residual;
    result.ForwardIterations = forward.Iterations;
    std::vector<int> samples;
    const int sampleCount = std::min(64, int(p.Faces.rows()));
    samples.reserve(sampleCount);
    for (int i = 0; i < sampleCount; ++i) samples.push_back(int(int64_t(i) * (p.Faces.rows() - 1) / (sampleCount - 1)));
    result.IndependentRows = sampleCount;
    Vector sampledIncident(sampleCount);
    for (int i = 0; i < sampleCount; ++i) sampledIncident[i] = incident[samples[i]];
    result.IndependentForwardResidual = (ReferenceRows(p, samples, true) * forward.Value + sampledIncident).norm() / sampledIncident.norm();
    if (!std::isfinite(result.IndependentForwardResidual) || result.IndependentForwardResidual > 1e-4) throw std::runtime_error("BEM forward solve failed independent FP64 sampled residual");
    result.ScatteredPressure = listener.Apply(forward.Value);
    const auto power = result.ScatteredPressure.cwiseAbs2().eval();
    const double sum = power.sum(), squared = power.squaredNorm();
    if (!(squared > 0.) || !std::isfinite(squared)) throw std::runtime_error("BEM diffusion undefined for zero/nonfinite pressure");
    result.Diffusion = (sum * sum - squared) / ((power.size() - 1) * squared);
    result.SolveSeconds = seconds(solveStart);
    if (!p.HeightVertices.empty()) {
        const auto adjointStart = Clock::now();
        const Vector vf = PressureSensitivity(result.ScatteredPressure), rhs = listener.Apply(vf, true);
        const std::function<Vector(const Vector &)> preconditionTranspose = options.Precondition == Preconditioner::None ? std::function<Vector(const Vector &)>{} : [&](const Vector &v) { return boundary.Precondition(v, true); };
        const auto adjoint = detail::Gmres([&](const Vector &v) { return boundary.Apply(v, true); }, rhs, options, preconditionTranspose);
        const Vector referenceRhs = ReferenceRows(p, samples, false, true) * vf;
        result.IndependentAdjointResidual = (ReferenceRows(p, samples, true, true) * adjoint.Value - referenceRhs).norm() / std::max(referenceRhs.norm(), 1e-300);
        if (!std::isfinite(result.IndependentAdjointResidual) || result.IndependentAdjointResidual > 1e-4) throw std::runtime_error("BEM adjoint solve failed independent FP64 sampled residual");
        result.AdjointResidual = adjoint.Residual;
        result.AdjointIterations = adjoint.Iterations;
        result.SolveSeconds += seconds(adjointStart);
        const auto derivativeStart = Clock::now();
        result.HeightGradient = MetalGradient(p, result.SurfacePressure, vf, adjoint.Value);
        result.DerivativeSeconds = seconds(derivativeStart);
        if (!result.HeightGradient.allFinite()) throw std::runtime_error("Nonfinite BEM height gradient");
    }
    return result;
}
} // namespace bem
