#include "Porous.h"
#include "RoomScene.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <unordered_map>

namespace Porous {
namespace {
constexpr uint16_t AllLinks = (1 << 12) - 1;

int ShellCount(const RoomScene &s, int i) {
    const int x = i / (s.Ny * s.Nz), y = i / s.Nz % s.Ny, z = i % s.Nz;
    return int(x == 1 || x == s.Nx - 2) + int(y == 1) + int(z == 1 || z == s.Nz - 2);
}

std::vector<uint16_t> Adjacencies(const RoomScene &s) {
    if (s.BnIxyz.size() != s.AdjBn.size() || s.BnIxyz.size() != s.MatBn.size())
        throw std::runtime_error("Porous scene boundary arrays disagree");
    std::vector<uint16_t> result(size_t(s.NumNodes()), AllLinks);
    for (size_t k = 0; k < s.BnIxyz.size(); ++k) {
        const int i = s.BnIxyz[k];
        if (!IsInterior(s, i) || (s.AdjBn[k] & ~AllLinks))
            throw std::runtime_error("Invalid FCC porous boundary adjacency");
        result[size_t(i)] = s.AdjBn[k];
    }
    return result;
}

void CheckScene(const RoomScene &s) {
    if (!s.Fcc || s.Nx < 6 || s.Ny < 6 || s.Nz < 6)
        throw std::runtime_error("Porous volumes require an FCC room grid with its halo");
    if (int64_t(s.Nx) * s.Ny * s.Nz > std::numeric_limits<int>::max())
        throw std::runtime_error("Porous grid exceeds 32-bit node indexing");
    if (!(s.H > 0 && s.Ts > 0 && s.C > 0 && s.Rho > 0) ||
        !std::isfinite(s.H + s.Ts + s.C + s.Rho) || s.C * s.Ts / s.H > 1 + 1e-12)
        throw std::runtime_error("Invalid porous grid spacing, material constants, or Courant number");
    const double courant = s.C * s.Ts / s.H;
    if (!std::isfinite(s.L + s.L2) || std::abs(s.L - courant) > 1e-8 ||
        std::abs(s.L2 - courant * courant) > 1e-8)
        throw std::runtime_error("Porous grid Courant metadata disagrees with its spacing and time step");
}

// Trapezoidal ADE: the state is centered on the input, while its average with
// the old state supplies the constitutive response over this time step.
double Response(const Rational &r, double input, double dt, double *const state) {
    double response = r.Constant * input;
    for (size_t i = 0; i < r.Poles.size(); ++i) {
        const auto &p = r.Poles[i];
        const double old = state[i], denominator = 2 + dt * p.Rate;
        const double next = ((2 - dt * p.Rate) * old + 2 * dt * input) / denominator;
        response += .5 * p.Residue * (old + next);
        state[i] = next;
    }
    return response;
}

void CheckRational(const Rational &r) {
    if (!(r.Constant > 0) || !std::isfinite(r.Constant))
        throw std::runtime_error("Invalid porous constitutive response");
    for (const auto &p : r.Poles)
        if (!(p.Rate > 0) || !std::isfinite(p.Rate) || !std::isfinite(p.Residue))
            throw std::runtime_error("Porous constitutive poles must be finite and strictly stable");
}
} // namespace

bool IsInterior(const RoomScene &s, int node) {
    if (node < 0 || node >= s.NumNodes()) return false;
    const int x = node / (s.Ny * s.Nz), y = node / s.Nz % s.Ny, z = node % s.Nz;
    return x > 0 && x < s.Nx - 1 && y > 0 && y < s.Ny - 1 && z > 0 && z < s.Nz - 1;
}

int UnfoldedY(const RoomScene &s, int node) {
    const int x = node / (s.Ny * s.Nz), y = node / s.Nz % s.Ny, z = node % s.Nz;
    return s.Fcc && ((x + y + z) & 1) ? s.NyUnfolded() - 1 - y : y;
}

std::array<int, 12> FccNeighbours(const RoomScene &s, int node) {
    const int x = node / (s.Ny * s.Nz), y = node / s.Nz % s.Ny, z = node % s.Nz;
    const int stride = s.Ny * s.Nz;
    const int xp = x == s.Nx - 2 ? -stride : stride, xm = x == 1 ? stride : -stride;
    const int yp = y == s.Ny - 2 ? 0 : s.Nz, ym = y == 1 ? s.Nz : -s.Nz;
    const int zp = z == s.Nz - 2 ? -1 : 1, zm = z == 1 ? 1 : -1;
    return {node + xp + yp, node + xm + ym, node + yp + zp, node + ym + zm, node + xp + zp, node + xm + zm, node + xp + ym, node + xm + yp, node + yp + zm, node + ym + zp, node + xp + zm, node + xm + zp};
}

std::shared_ptr<Prepared> Prepare(const RoomScene &s, const std::vector<int> &labels, const std::vector<Material> &materials) {
    CheckScene(s);
    if (labels.size() != size_t(s.NumNodes()))
        throw std::runtime_error("Porous labels must cover the folded room grid");
    auto result = std::make_shared<Prepared>();
    result->Materials.push_back(AirMaterial(s.C, s.Rho));
    result->Materials.insert(result->Materials.end(), materials.begin(), materials.end());
    for (const auto &material : result->Materials) {
        CheckRational(material.BulkModulus);
        CheckRational(material.InverseDensity);
        CheckRational(material.WaveSpeedSquared);
        const double speed2 = std::max(material.WaveSpeedSquared.Constant, material.BulkModulus.Constant * material.InverseDensity.Constant);
        if (s.Ts * std::sqrt(speed2) > s.H * (1 + 1e-12))
            throw std::runtime_error("Porous high-frequency wave speed exceeds the FCC Courant limit");
    }
    const auto adjacency = Adjacencies(s);
    // A full-mask rigid boundary on the outer shell deliberately selects the
    // existing mirrored stencil instead of its absorbing correction. This is
    // useful for exact symmetry-plane fixtures; ordinary padded rooms keep
    // their dropped-link walls and may not reach the absorbing shell.
    std::vector<bool> mirror_shell(labels.size());
    for (size_t k = 0; k < s.BnIxyz.size(); ++k)
        if (s.MatBn[k] < 0 && s.AdjBn[k] == AllLinks)
            mirror_shell[size_t(s.BnIxyz[k])] = true;
    std::vector<uint8_t> affected(labels.size());
    for (int i = 0; i < s.NumNodes(); ++i) {
        if (labels[size_t(i)] < -1 || labels[size_t(i)] >= int(materials.size()))
            throw std::runtime_error("Porous node refers to an undefined material");
        if (!IsInterior(s, i)) {
            if (labels[size_t(i)] != -1) throw std::runtime_error("Porous volume enters the held-zero grid halo");
            continue;
        }
        const auto neighbours = FccNeighbours(s, i);
        bool interface = false;
        for (int b = 0; b < 12; ++b)
            if ((adjacency[size_t(i)] >> b & 1) && labels[size_t(i)] != labels[size_t(neighbours[b])])
                interface = true;
        if (interface) {
            result->Interface.push_back({i, labels[size_t(i)] + 1, {}});
            affected[size_t(i)] = 1;
        } else if (labels[size_t(i)] >= 0) {
            result->Interior.push_back({i, labels[size_t(i)] + 1, adjacency[size_t(i)]});
            affected[size_t(i)] = 1;
        }
        if (affected[size_t(i)] && ShellCount(s, i) && !mirror_shell[size_t(i)])
            throw std::runtime_error("Porous volume/interface touches the outer absorbing shell; enlarge the grid padding");
    }
    for (size_t k = 0; k < s.BnIxyz.size(); ++k)
        if (s.MatBn[k] >= 0 && affected[size_t(s.BnIxyz[k])])
            throw std::runtime_error("A locally reacting wall overlaps a porous volume/interface");
    for (int const i : s.InIxyz)
        if (i < 0 || i >= s.NumNodes() || affected[size_t(i)])
            throw std::runtime_error("Room sources must be inside scalar air, away from porous interfaces");

    std::unordered_map<uint64_t, int> link_indices;
    std::map<std::pair<int, int>, int> link_materials;
    for (auto &row : result->Interface) {
        const auto neighbours = FccNeighbours(s, row.Node);
        for (int b = 0; b < 12; ++b) {
            if (!(adjacency[size_t(row.Node)] >> b & 1)) continue;
            const int neighbour = neighbours[b];
            if (ShellCount(s, neighbour) && !mirror_shell[size_t(neighbour)])
                throw std::runtime_error("Porous staggered links reach the absorbing shell; enlarge the grid padding");
            // Folded y reverses the orientation of some stencil bits. Endpoint
            // pairs, rather than a presumed opposite bit, identify shared links.
            const auto reverse = FccNeighbours(s, neighbour);
            bool reciprocal = false;
            for (int r = 0; r < 12; ++r)
                reciprocal |= reverse[r] == row.Node && (adjacency[size_t(neighbour)] >> r & 1);
            if (!reciprocal)
                throw std::runtime_error("Porous interface has a nonreciprocal room adjacency link");
            const int a = std::min(row.Node, neighbour), c = std::max(row.Node, neighbour);
            const uint64_t key = (uint64_t(uint32_t(a)) << 32) | uint32_t(c);
            auto [entry, inserted] = link_indices.emplace(key, int(result->Links.size()));
            if (inserted) {
                const int ma = labels[size_t(a)] + 1, mb = labels[size_t(c)] + 1;
                const auto pair = std::minmax(ma, mb);
                auto [model, new_model] = link_materials.emplace(pair, int(result->LinkInverseDensity.size()));
                if (new_model) {
                    const auto &am = result->Materials[size_t(ma)], &bm = result->Materials[size_t(mb)];
                    result->LinkInverseDensity.push_back(ma == mb ? am.InverseDensity : PrepareLinkInverseDensity(am, bm, s.Rho));
                    CheckRational(result->LinkInverseDensity.back());
                }
                result->Links.push_back({a, c, model->second});
            }
            row.Links[b] = (row.Node == a ? 1 : -1) * (entry->second + 1);
        }
    }
    return result;
}

void Cpu::Init(const RoomScene &s) {
    CheckScene(s);
    if (s.NumLossy || std::any_of(s.MatBn.begin(), s.MatBn.end(), [](int m) { return m >= 0; }))
        throw std::runtime_error("The porous FP64 diagnostic reference requires rigid walls");
    if (s.Nt < 0 || s.InSigs.size() != size_t(s.Nt) * s.InIxyz.size())
        throw std::runtime_error("Porous reference source dimensions disagree");
    for (const auto *indices : {&s.InIxyz, &s.OutIxyz})
        for (int const node : *indices)
            if (!IsInterior(s, node))
                throw std::runtime_error("Porous reference source/receiver is outside the active grid");
    Scene = &s;
    Data = s.Porous ? s.Porous : std::make_shared<Prepared>();
    Adjacency = Adjacencies(s);
    Boundary.assign(size_t(s.NumNodes()), 0);
    for (int const node : s.BnIxyz) Boundary[size_t(node)] = 1;
    for (const auto &r : Data->Interior) Boundary[size_t(r.Node)] |= 2;
    for (const auto &r : Data->Interface) Boundary[size_t(r.Node)] |= 2;
    Current.assign(size_t(s.NumNodes()), 0);
    Previous = Current;
    Velocity.assign(Data->Links.size(), 0);
    ReceiverOutput.assign(size_t(s.Nt) * s.OutIxyz.size(), 0);
    auto const allocate = [](const auto &rows, auto &&response, auto &offset, auto &state) {
        offset.assign(rows.size() + 1, 0);
        for (size_t i = 0; i < rows.size(); ++i) offset[i + 1] = offset[i] + response(rows[i]).Poles.size();
        // Keep a valid pointer even for constitutive responses with no poles.
        state.assign(std::max<size_t>(1, offset.back()), 0);
    };
    allocate(Data->Interior, [&](const auto &r) -> const Rational & { return Data->Materials[r.Material].WaveSpeedSquared; }, InteriorOffset, InteriorState);
    allocate(Data->Interface, [&](const auto &r) -> const Rational & { return Data->Materials[r.Material].BulkModulus; }, PressureOffset, PressureState);
    allocate(Data->Links, [&](const auto &r) -> const Rational & { return Data->LinkInverseDensity[r.Material]; }, VelocityOffset, VelocityState);
    Step = 0;
}

void Cpu::RunSteps(int count) {
    if (!Scene || count < 0 || count > Scene->Nt - Step)
        throw std::runtime_error("Porous reference step count exceeds scene duration");
    const auto &s = *Scene;
    const double dt = s.Ts, laplace_scale = 1 / (4 * s.H * s.H);
    const double gradient_scale = 1 / (std::numbers::sqrt2 * s.H);
    const double divergence_scale = .5 * gradient_scale;
    const auto laplacian = [&](int node, uint32_t adjacency) {
        double value = 0;
        const auto neighbours = FccNeighbours(s, node);
        for (int b = 0; b < 12; ++b)
            if (adjacency >> b & 1) value += Current[neighbours[b]] - Current[node];
        return value;
    };
    // All updates read Current and write Previous until the final swap.
    for (int n = 0; n < count; ++n) {
        for (size_t j = 0; j < s.OutIxyz.size(); ++j)
            ReceiverOutput[size_t(Step) * s.OutIxyz.size() + j] = Current[size_t(s.OutIxyz[j])];
        for (int x = 1; x < s.Nx - 1; ++x)
            for (int y = 1; y < s.Ny - 1; ++y)
                for (int z = 1; z < s.Nz - 1; ++z) {
                    const int i = (x * s.Ny + y) * s.Nz + z;
                    if (Boundary[size_t(i)] & 2) continue;
                    const double old = Previous[size_t(i)], lq = s.C * dt / s.H * ShellCount(s, i);
                    double value = 2 * Current[size_t(i)] - old + dt * dt * s.C * s.C * laplace_scale * laplacian(i, Adjacency[i]);
                    if (!(Boundary[size_t(i)] & 1)) value = (value + lq * old) / (1 + lq);
                    Previous[size_t(i)] = value;
                }
        for (size_t j = 0; j < Data->Interior.size(); ++j) {
            const auto &r = Data->Interior[j];
            const double acceleration = Response(Data->Materials[r.Material].WaveSpeedSquared, laplace_scale * laplacian(r.Node, r.Adjacency), dt, InteriorState.data() + InteriorOffset[j]);
            Previous[size_t(r.Node)] = 2 * Current[size_t(r.Node)] - Previous[size_t(r.Node)] + dt * dt * acceleration;
        }
        for (size_t j = 0; j < Data->Links.size(); ++j) {
            const auto &r = Data->Links[j];
            const double gradient = gradient_scale * (Current[size_t(r.B)] - Current[size_t(r.A)]);
            Velocity[j] -= dt * Response(Data->LinkInverseDensity[r.Material], gradient, dt, VelocityState.data() + VelocityOffset[j]);
        }
        for (size_t j = 0; j < Data->Interface.size(); ++j) {
            const auto &r = Data->Interface[j];
            double divergence = 0;
            for (int const link : r.Links)
                if (link) divergence += (link > 0 ? 1 : -1) * Velocity[size_t(std::abs(link) - 1)];
            Previous[size_t(r.Node)] = Current[size_t(r.Node)] -
                dt * Response(Data->Materials[r.Material].BulkModulus, divergence_scale * divergence, dt, PressureState.data() + PressureOffset[j]);
        }
        for (size_t j = 0; j < s.InIxyz.size(); ++j)
            Previous[size_t(s.InIxyz[j])] += s.InSigs[j * size_t(s.Nt) + size_t(Step)];
        Current.swap(Previous);
        ++Step;
    }
}
} // namespace Porous
