// Included privately by the production runner and RoomTest.
// Smits & Bilbao 2025 scheme parameters and Table II Jacobi counts.
struct ImplicitScheme {
    const char *Name;
    double Lex[3], Lim[3], Lambda, Eps;
    int Sweeps;
};

constexpr ImplicitScheme Implicit2Pct{"IM 2%", {0.390, 0.610, 0.000}, {0., 2.09e-3, 41.17e-3}, 0.845, .02, 7};
constexpr ImplicitScheme Implicit1Pct{"IM 1%", {0.404, 0.553, 0.043}, {0., 9.75e-3, 28.37e-3}, 0.840, .01, 7};
constexpr ImplicitScheme ImplicitHalfPct{"IM 0.5%", {0.400, 0.580, 0.020}, {0., 0.91e-3, 29.95e-3}, 0.850, .005, 6};

// Room-scale fitted drop-ghost wall offset; ImplicitModeRung measures size dependence.
constexpr double ImplicitWallOffset = 0.44;

// Paper Eq. 16 over the Table I corners. Recompute because Table II rounds lambda_max; the
// rounded IM 0.5% value exceeds its 0.8494 limit.
double ImplicitCourantLimit(const ImplicitScheme &s) {
    static constexpr double L[3][3] = {{-4., -4., -4.}, {-8., -4., 0.}, {-12., 0., -4.}};
    double limit = 1e30;
    for (const auto &lb : L) {
        double ex = 0., im = 1.;
        for (int r = 0; r < 3; ++r) {
            ex -= s.Lex[r] * lb[r];
            im += s.Lim[r] * lb[r];
        }
        if (ex > 0.) limit = std::min(limit, std::sqrt(4. * im / ex));
    }
    return limit;
}

// Narrow once so diagnostics use the exact coefficients passed to Metal.
RoomImplicitParams ImplicitCoefficients(const ImplicitScheme &s, double lambda, int nx, int ny, int nz) {
    const double e[3] = {s.Lex[0], s.Lex[1] / 4., s.Lex[2] / 4.};
    const double n[3] = {s.Lim[0], s.Lim[1] / 4., s.Lim[2] / 4.};
    const double ediag = 6. * s.Lex[0] + 3. * s.Lex[1] + 2. * s.Lex[2];
    const double jd = 1. - (6. * s.Lim[0] + 3. * s.Lim[1] + 2. * s.Lim[2]);
    const double l2 = lambda * lambda;
    return {nx, ny, nz, 0, 0, 0, 0, float((2. * n[0] + l2 * e[0]) / jd), float((2. * n[1] + l2 * e[1]) / jd), float((2. * n[2] + l2 * e[2]) / jd), float((2. * jd - l2 * ediag) / jd), float(n[0] / jd), float(n[1] / jd), float(n[2] / jd)};
}

// Plane-wave symbol of the exact float coefficients used by the kernels.
double ImplicitTwoCosOmegaT(const RoomImplicitParams &p, double kx, double ky, double kz) {
    const double cx = std::cos(kx), cy = std::cos(ky), cz = std::cos(kz);
    const double s1 = 2. * (cx + cy + cz), s2 = 4. * (cx * cy + cx * cz + cy * cz), s3 = 8. * cx * cy * cz;
    const double off = p.Q1 * s1 + p.Q2 * s2 + p.Q3 * s3;
    return 1. + ((p.R1 * s1 + p.R2 * s2 + p.R3 * s3 + p.Rc) - (off + 1.)) / (1. + off);
}

// Undivided off-diagonal coefficient for a face, edge, or corner neighbor.
double ImplicitNim(const ImplicitScheme &s, int r) { return r == 0 ? s.Lim[0] : s.Lim[r] / 4.; }
double ImplicitNex(const ImplicitScheme &s, int r) { return r == 0 ? s.Lex[0] : s.Lex[r] / 4.; }

// Precomputed mesh geometry consumed by the boundary operator.
struct ImplicitVoxels {
    int N[3]{0, 0, 0};
    std::vector<uint32_t> Blocked;
    std::vector<uint8_t> Filled;
    std::vector<float> Normals;
    std::vector<int8_t> Mat;
    size_t At(int ix, int iy, int iz) const { return (size_t(ix) * size_t(N[1]) + size_t(iy)) * size_t(N[2]) + size_t(iz); }
};

// Raster position in a 3x3x3 neighborhood with the center omitted.
constexpr int ImplicitLegBit(int dx, int dy, int dz) {
    const int i = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1);
    return i < 13 ? i : i - 1;
}

struct ImplicitLeg {
    int Dx, Dy, Dz, Class, Bit, OppositeBit;
};

constexpr std::array<ImplicitLeg, 26> ImplicitLegs = [] {
    std::array<ImplicitLeg, 26> legs{};
    size_t i = 0;
    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz) {
                const int squared_distance = dx * dx + dy * dy + dz * dz;
                if (squared_distance) legs[i++] = {dx, dy, dz, squared_distance - 1, ImplicitLegBit(dx, dy, dz), ImplicitLegBit(-dx, -dy, -dz)};
            }
    return legs;
}();

constexpr uint32_t ImplicitLegClassMask(int squared_distance) {
    uint32_t mask = 0;
    for (const auto &leg : ImplicitLegs)
        if (leg.Class == squared_distance - 1) mask |= 1u << leg.Bit;
    return mask;
}

constexpr uint32_t ImplicitAllLegs = (1u << 26) - 1;

// Closed-form geometries used to isolate boundary and voxelization behavior.
struct ImplicitShape {
    enum Kind { Array,
                Slanted,
                Sphere,
                Obstacle,
                Voxels };

    explicit ImplicitShape(const ImplicitVoxels &vox) : ImplicitShape(Voxels, vox.N[0], vox.N[1], vox.N[2]) { Vox = &vox; }

    ImplicitShape(Kind k, int nx, int ny, int nz, double angle = 0.) : K(k), N{nx, ny, nz}, Cos(std::cos(angle)), Sin(std::sin(angle)) {
        // Break octahedral symmetry so term-count diagnostics exercise general orientations.
        static constexpr double Break[3]{0.37, 0.19, 0.23};
        double fit = 1e30;
        for (int a = 0; a < 3; ++a) {
            Cen[a] = .5 * (N[a] - 1) + (k == Array ? 0. : Break[a]);
            fit = std::min(fit, std::min(Cen[a], N[a] - 1 - Cen[a]) + .5);
        }
        Rad = k == Obstacle ? .4 * fit : fit;
        Ext[2] = std::min(Cen[2], N[2] - 1 - Cen[2]) + .5;
        Ext[0] = Ext[1] = std::min(fit, Ext[2]) / (std::abs(Cos) + std::abs(Sin));
    }

    const char *Name() const {
        static const char *const Names[4]{"the array", "a yawed box", "a sphere", "a column in the array"};
        return Names[K];
    }

    void Local(const double p[3], double q[3]) const {
        q[0] = Cos * p[0] + Sin * p[1];
        q[1] = Cos * p[1] - Sin * p[0];
        q[2] = p[2];
    }

    double FromCentre(int ix, int iy, int iz, double p[3]) const {
        p[0] = ix - Cen[0];
        p[1] = iy - Cen[1];
        p[2] = iz - Cen[2];
        return std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    }

    bool Inside(int ix, int iy, int iz) const {
        if (ix < 0 || iy < 0 || iz < 0 || ix >= N[0] || iy >= N[1] || iz >= N[2]) return false;
        // The fill uses the same 26-leg connectivity as the operator.
        if (K == Voxels) return Vox->Filled[Vox->At(ix, iy, iz)] != 0;
        if (K == Array) return true;
        double p[3];
        const double radius = FromCentre(ix, iy, iz, p);
        if (K == Sphere) return radius <= Rad - .5;
        if (K == Obstacle) return radius >= Rad + .5; // and inside the array, tested above
        double q[3];
        Local(p, q);
        return std::abs(q[0]) <= Ext[0] - .5 && std::abs(q[1]) <= Ext[1] - .5 && std::abs(q[2]) <= Ext[2] - .5;
    }

    int Material(int ix, int iy, int iz) const {
        if (K != Voxels) return 0;
        return int(Vox->Mat[Vox->At(ix, iy, iz)]);
    }

    // True outward surface normal, or zero away from a wall.
    void Normal(int ix, int iy, int iz, double n[3]) const {
        n[0] = n[1] = n[2] = 0.;
        if (K == Voxels) {
            for (int a = 0; a < 3; ++a) n[a] = double(Vox->Normals[3 * Vox->At(ix, iy, iz) + size_t(a)]);
            return;
        }
        if (K == Array) {
            n[0] = double(ix == N[0] - 1) - double(ix == 0);
            n[1] = double(iy == N[1] - 1) - double(iy == 0);
            n[2] = double(iz == N[2] - 1) - double(iz == 0);
        } else {
            double p[3];
            const double radius = FromCentre(ix, iy, iz, p);
            if (K == Sphere) {
                for (int a = 0; a < 3; ++a) n[a] = p[a];
            } else if (K == Obstacle) {
                // The column's outward-from-room normal points into the obstacle.
                int face = 0;
                for (int a = 1; a < 3; ++a)
                    if (Wall(a) - std::abs(p[a]) < Wall(face) - std::abs(p[face])) face = a;
                if (radius - Rad < Wall(face) - std::abs(p[face])) {
                    for (int a = 0; a < 3; ++a) n[a] = -p[a];
                } else {
                    n[face] = p[face] < 0. ? -1. : 1.;
                }
            } else {
                double q[3];
                Local(p, q);
                int face = 0;
                for (int a = 1; a < 3; ++a)
                    if (Ext[a] - std::abs(q[a]) < Ext[face] - std::abs(q[face])) face = a;
                double m[3]{0., 0., 0.};
                m[face] = q[face] < 0. ? -1. : 1.;
                n[0] = Cos * m[0] - Sin * m[1];
                n[1] = Sin * m[0] + Cos * m[1];
                n[2] = m[2];
            }
        }
        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 0.)
            for (int a = 0; a < 3; ++a) n[a] /= len;
    }

    double Wall(int a) const { return std::min(Cen[a], N[a] - 1 - Cen[a]) + .5; }

    // Mesh crossings remain authoritative even when an open model leaks the fill.
    bool Blocked(int ix, int iy, int iz, int dx, int dy, int dz) const {
        if (K == Voxels) return (Vox->Blocked[Vox->At(ix, iy, iz)] & (1u << ImplicitLegBit(dx, dy, dz))) != 0;
        return !Inside(ix + dx, iy + dy, iz + dz);
    }

    void Counts(int ix, int iy, int iz, int k[3]) const {
        if (K == Voxels) {
            const uint32_t blocked = Vox->Blocked[Vox->At(ix, iy, iz)];
            static constexpr int Full[3]{6, 12, 8};
            static constexpr uint32_t Mask[3]{ImplicitLegClassMask(1), ImplicitLegClassMask(2), ImplicitLegClassMask(3)};
            for (int r = 0; r < 3; ++r) k[r] = Full[r] - std::popcount(blocked & Mask[r]);
            return;
        }
        if (K == Array) {
            const int a = int(ix > 0) + int(ix < N[0] - 1), b = int(iy > 0) + int(iy < N[1] - 1), c = int(iz > 0) + int(iz < N[2] - 1);
            k[0] = a + b + c;
            k[1] = a * b + a * c + b * c;
            k[2] = a * b * c;
            return;
        }
        k[0] = k[1] = k[2] = 0;
        for (const auto &leg : ImplicitLegs)
            if (Inside(ix + leg.Dx, iy + leg.Dy, iz + leg.Dz)) ++k[leg.Class];
    }

    uint32_t KeptLegs(int ix, int iy, int iz) const {
        if (K == Voxels) return ~Vox->Blocked[Vox->At(ix, iy, iz)] & ImplicitAllLegs;
        uint32_t bits = 0;
        for (const auto &leg : ImplicitLegs)
            if (!Blocked(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz)) bits |= 1u << leg.Bit;
        return bits;
    }

    Kind K;
    const ImplicitVoxels *Vox{nullptr};
    int N[3]; // NOLINT(modernize-use-default-member-init) the constructor's arguments are where it comes from
    double Cos, Sin;
    double Cen[3], Ext[3], Rad;
};

// Computes kept counts and beta_i over ghost offsets using the true surface normal.
double ImplicitWallSample(const ImplicitShape &shape, int ix, int iy, int iz, const double nex[3], int k[3], bool compensated) {
    double n[3];
    shape.Normal(ix, iy, iz, n);
    k[0] = k[1] = k[2] = 0;
    double beta = 0., faces = 0., projected = 0.;
    for (const auto &leg : ImplicitLegs) {
        if (!shape.Blocked(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz)) {
            ++k[leg.Class];
            continue;
        }
        const double projection = n[0] * leg.Dx + n[1] * leg.Dy + n[2] * leg.Dz;
        beta += nex[leg.Class] * projection;
        if (leg.Class == 0) {
            ++faces;
            projected += projection;
        }
    }
    // Uncompensated beta restores the staircased rather than projected surface area.
    if (!compensated && projected > 0.) beta *= faces / projected;
    // Opposing walls in sub-cell gaps can make beta negative; clamp them rigid to preserve
    // the passive-boundary energy condition.
    return std::max(beta, 0.);
}

// Least-squares slope from `from`, or infinity with fewer than four samples.
double ImplicitSlope(const std::vector<double> &x, const std::vector<double> &y, size_t from = 0) {
    if (x.size() < from + 4) return std::numeric_limits<double>::infinity();
    const double count = double(x.size() - from);
    double sx = 0., sy = 0., sxx = 0., sxy = 0.;
    for (size_t i = from; i < x.size(); ++i) {
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
    }
    return (count * sxy - sx * sy) / (count * sxx - sx * sx);
}

struct ImplicitWall {
    double Gamma{0.};
    double Ts{0.};
    std::vector<double> Def; // (D, E, F) triples
    std::vector<double> GammaByMaterial;
    int Branches() const { return int(Def.size()) / 3; }
    bool HasRealAdmittance() const {
        if (Gamma != 0.) return true;
        return std::any_of(GammaByMaterial.begin(), GammaByMaterial.end(), [](double gamma) { return gamma != 0.; });
    }
    double RealAdmittance(int material) const {
        if (material >= 0 && material < int(GammaByMaterial.size())) return GammaByMaterial[size_t(material)];
        return Gamma;
    }
};

struct ImplicitBox {
    // A rigid box adds a held-zero halo; wall terms restore the diagonal lost with each ghost.
    // Nonnegative admittance remains diagonal and therefore does not tighten the Courant bound.
    ImplicitBox(const ImplicitScheme &s, int nx, int ny, int nz, bool rigid = false, double lambda = 0., ImplicitWall wall = {}, ImplicitShape::Kind kind = ImplicitShape::Array, double angle = 0., bool compensated = true, const ImplicitVoxels *vox = nullptr)
        : Scheme(s), Lambda(lambda > 0. ? lambda : std::min(s.Lambda, ImplicitCourantLimit(s))), Wall(std::move(wall)), Compensated(compensated),
          Shape(vox ? ImplicitShape(*vox) : ImplicitShape(kind, nx, ny, nz, angle)), Pad(rigid ? 1 : 0),
          Nx(nx), Ny(ny), Nz(nz), Gx(nx + 2 * Pad), Gy(ny + 2 * Pad), Gz(nz + 2 * Pad),
          P(ImplicitCoefficients(s, Lambda, Gx, Gy, Gz)), Nodes(size_t(Gx) * Gy * Gz), Room(size_t(nx) * ny * nz) {
        for (auto *b : {&Prev, &Cur, &Bp, &Tmp}) b->ResizeZeroed(Nodes * sizeof(float));
        auto &ctx = MetalContext::Get();
        Rhs = ctx.RoomPipeline(rigid ? "RoomImplicitRhsNoWall" : "RoomImplicitRhs");
        Jacobi = ctx.RoomPipeline(rigid ? "RoomImplicitJacobiNoWall" : "RoomImplicitJacobi");
        Stream = ctx.RoomPipeline(rigid ? "RoomImplicitStreamWall" : "RoomImplicitStream");
        Threads = {32, 4, 4};
        Tiles = {(uint32_t(nz) + Threads.x - 1) / Threads.x, (uint32_t(ny) + Threads.y - 1) / Threads.y, (uint32_t(nx) + Threads.z - 1) / Threads.z};
        if (!rigid) return;

        RhsFused = ctx.RoomPipeline("RoomImplicitRhsWall");
        JacobiFused = ctx.RoomPipeline("RoomImplicitJacobiWall");

        if (Wall.Branches()) {
            RoomScene scene;
            scene.Ts = Wall.Ts;
            scene.MatBranches = {Wall.Branches()};
            scene.MatDef = Wall.Def;
            Mats = RoomMaterialCoefficients(scene);
            WallRhsPso = ctx.RoomPipeline("RoomImplicitWallRhs");
            WallStepPso = ctx.RoomPipeline("RoomImplicitWallStep");
        }
        std::vector<RoomImplicitWall> lossy;
        const bool has_wall_admittance = HasWallAdmittance();

        // Curved production rooms exceed 65,535 exact terms, so term indices are uint32_t.
        std::vector<RoomImplicitTerm> terms{{0.f, 0.f, 1.f}, {0.f, 0.f, 0.f}};
        struct TermKey {
            uint32_t Cn, Cp, Fd;
            bool operator==(const TermKey &) const = default;
        };
        struct TermHash {
            size_t operator()(const TermKey &k) const {
                size_t h = size_t(k.Cn) * 0x9e3779b1u;
                h ^= size_t(k.Cp) + 0x9e3779b9u + (h << 6) + (h >> 2);
                h ^= size_t(k.Fd) + 0x9e3779b9u + (h << 6) + (h >> 2);
                return h;
            }
        };
        const auto key = [](const RoomImplicitTerm &t) { return TermKey{std::bit_cast<uint32_t>(t.Cn), std::bit_cast<uint32_t>(t.Cp), std::bit_cast<uint32_t>(t.Fd)}; };
        std::unordered_map<TermKey, uint32_t, TermHash> term_index{{key(terms[0]), 0u}, {key(terms[1]), 1u}};
        std::vector<RoomImplicitBlockEntry> blocks((Nodes + RoomBnBlock - 1) / RoomBnBlock, RoomImplicitBlockEntry{});
        std::vector<uint32_t> code;
        std::vector<uint32_t> keep;
        for (int ix = 0; ix < nx; ++ix) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int iz = 0; iz < nz; ++iz) {
                    const size_t ii = Index(ix, iy, iz);
                    if (!Shape.Inside(ix, iy, iz)) {
                        blocks[ii / RoomBnBlock].Outside |= 1u << (ii % RoomBnBlock);
                        continue;
                    }
                    ++InsideNodes;
                    int k[3];
                    Shape.Counts(ix, iy, iz, k);
                    if (k[0] == 6 && k[1] == 12 && k[2] == 8) continue;
                    const double scalar = WallScalar(ix, iy, iz);
                    const double beta = scalar != 0. || Wall.Branches() ? Beta(ix, iy, iz) : 0.;
                    const double g = scalar == 0. ? 0. : .5 * Lambda * beta * scalar;
                    if (has_wall_admittance) WallG.emplace_back(int(ii), g);
                    if (Wall.Branches()) lossy.push_back({int(ii), 0, float(Lambda * beta / Diagonal())});
                    const RoomImplicitTerm t = WallTerms(k, g);
                    const auto [it, inserted] = term_index.emplace(key(t), uint32_t(terms.size()));
                    if (inserted) terms.push_back(t);
                    blocks[ii / RoomBnBlock].Wall |= 1u << (ii % RoomBnBlock);
                    code.push_back(it->second);
                    keep.push_back(Shape.KeptLegs(ix, iy, iz));
                    ++WallNodes;
                }
            }
        }
        uint32_t rank = 0;
        for (auto &b : blocks) {
            b.Rank = rank;
            rank += uint32_t(std::popcount(b.Wall));
        }
        Blocks.ResizeZeroed(blocks.size() * sizeof(RoomImplicitBlockEntry));
        Blocks.Upload(blocks.data(), blocks.size() * sizeof(RoomImplicitBlockEntry));
        Keep.ResizeZeroed(keep.size() * sizeof(uint32_t));
        Keep.Upload(keep.data(), keep.size() * sizeof(uint32_t));

        Codes = int(terms.size());
        Code.ResizeZeroed(code.size() * sizeof(uint32_t));
        Code.Upload(code.data(), code.size() * sizeof(uint32_t));
        Terms.ResizeZeroed(terms.size() * sizeof(RoomImplicitTerm));
        Terms.Upload(terms.data(), terms.size() * sizeof(RoomImplicitTerm));
        if (lossy.empty()) return;

        P.Nw = int(lossy.size());
        WallList.ResizeZeroed(lossy.size() * sizeof(RoomImplicitWall));
        WallList.Upload(lossy.data(), lossy.size() * sizeof(RoomImplicitWall));
        const int mb = Wall.Branches();
        MatMb.ResizeZeroed(sizeof(int));
        MatMb.Upload(&mb, sizeof(int));
        MatQuads.ResizeZeroed(Mats.Quads.size() * sizeof(RoomMatQuad));
        MatQuads.Upload(Mats.Quads.data(), Mats.Quads.size() * sizeof(RoomMatQuad));
        for (auto *b : {&Vh, &Gh}) b->ResizeZeroed(size_t(mb) * lossy.size() * sizeof(float));
        Ub.ResizeZeroed(lossy.size() * sizeof(float));
        WallThreads = {64, 1, 1};
        WallTiles = {(uint32_t(P.Nw) + WallThreads.x - 1) / WallThreads.x, 1, 1};
    }

    bool HasWallAdmittance() const { return Wall.Branches() || Wall.HasRealAdmittance(); }
    double WallScalar(int ix, int iy, int iz) const {
        return Wall.Branches() ? double(Mats.Beta[0]) : Wall.RealAdmittance(Shape.Material(ix, iy, iz));
    }

    double Nim(int r) const { return ImplicitNim(Scheme, r); }
    double Nex(int r) const { return ImplicitNex(Scheme, r); }
    double Diagonal(const int k[3] = nullptr) const {
        static constexpr int Full[3]{6, 12, 8};
        const int *c = k ? k : Full;
        return 1. - (Nim(0) * c[0] + Nim(1) * c[1] + Nim(2) * c[2]);
    }
    double ExDiagonal(const int k[3] = nullptr) const {
        static constexpr int Full[3]{6, 12, 8};
        const int *c = k ? k : Full;
        return Nex(0) * c[0] + Nex(1) * c[1] + Nex(2) * c[2];
    }

    size_t Index(int ix, int iy, int iz) const { return (size_t(ix + Pad) * Gy + size_t(iy + Pad)) * Gz + size_t(iz + Pad); }

    // Corrects the uniform RHS and Jacobi diagonal for the neighbors a wall drops.
    RoomImplicitTerm WallTerms(const int k[3], double g) const {
        const double d = Diagonal(k), ex = ExDiagonal(k), d0 = Diagonal();
        return {float((2. * d - Lambda * Lambda * ex) / d0 - double(P.Rc)), float(1. - (d - g) / d0), float(d0 / (d + g))};
    }
    RoomImplicitTerm WallTerms(int ix, int iy, int iz) const {
        if (!Shape.Inside(ix, iy, iz)) return {0.f, 0.f, 0.f};
        int k[3];
        Shape.Counts(ix, iy, iz, k);
        return WallTerms(k, NodeG(ix, iy, iz));
    }

    double Beta(int ix, int iy, int iz) const {
        const double nex[3]{Nex(0), Nex(1), Nex(2)};
        int k[3];
        return ImplicitWallSample(Shape, ix, iy, iz, nex, k, Compensated);
    }
    double NodeG(int ix, int iy, int iz) const {
        const double scalar = WallScalar(ix, iy, iz);
        return scalar == 0. ? 0. : .5 * Lambda * Beta(ix, iy, iz) * scalar;
    }

    // Independent levels avoid the amplitude bias produced by seeding a mode at rest.
    template<typename F, typename G> void Seed(F &&cur, G &&prev) {
        std::vector<float> a(Nodes, 0.f), b(Nodes, 0.f);
        for (int ix = 0; ix < Nx; ++ix)
            for (int iy = 0; iy < Ny; ++iy)
                for (int iz = 0; iz < Nz; ++iz) {
                    if (!Shape.Inside(ix, iy, iz)) continue;
                    a[Index(ix, iy, iz)] = float(cur(ix, iy, iz));
                    b[Index(ix, iy, iz)] = float(prev(ix, iy, iz));
                }
        Cur.Upload(a.data(), a.size() * sizeof(float));
        Prev.Upload(b.data(), b.size() * sizeof(float));
        if (!P.Nw) return;
        // Reset branch history consistently with the new previous level.
        const size_t branch_bytes = size_t(Wall.Branches()) * size_t(P.Nw) * sizeof(float);
        Vh.Zero(branch_bytes);
        Gh.Zero(branch_bytes);
        std::vector<float> behind(size_t(P.Nw));
        const RoomImplicitWall *list = WallList.As<RoomImplicitWall>();
        for (int i = 0; i < P.Nw; ++i) behind[size_t(i)] = b[size_t(list[i].Ixyz)];
        Ub.Upload(behind.data(), behind.size() * sizeof(float));
    }

    // Four grids implement the paper's S=4 state with handle swaps instead of copies.
    // `wall=false` exists only to isolate boundary cost on the same geometry.
    void Step(bool wall = true, bool project = true) {
        if (project && ProjectionEvery && !wall) throw std::runtime_error("mean projection needs the wall path");
        auto &ctx = MetalContext::Get();
        const bool fused = WallNodes > 0 && wall;
        if (fused) ctx.Dispatch(RhsFused, Tiles, Threads, {&Bp, &Cur, &Prev, &Blocks, &Code, &Terms, &Keep}, &P, sizeof P);
        else ctx.Dispatch(Rhs, Tiles, Threads, {&Bp, &Cur, &Prev}, &P, sizeof P);
        if (fused && P.Nw) ctx.Dispatch(WallRhsPso, WallTiles, WallThreads, {&Bp, &Vh, &Gh, &WallList, &MatMb, &MatQuads}, &P, sizeof P);
        const GpuBuffer *src = &Cur, *dst = &Prev;
        for (int m = 0; m < Scheme.Sweeps; ++m) {
            if (fused) ctx.Dispatch(JacobiFused, Tiles, Threads, {dst, src, &Bp, &Blocks, &Code, &Terms, &Keep}, &P, sizeof P);
            else ctx.Dispatch(Jacobi, Tiles, Threads, {dst, src, &Bp}, &P, sizeof P);
            src = dst;
            dst = (dst == &Prev) ? &Tmp : &Prev;
        }
        if (src != &Prev) {
            Prev.Swap(Tmp);
        }
        Prev.Swap(Cur);
        if (fused && P.Nw) ctx.Dispatch(WallStepPso, WallTiles, WallThreads, {&Cur, &Prev, &Ub, &Vh, &Gh, &WallList, &MatMb, &MatQuads}, &P, sizeof P);
        if (project) AdvanceMeanProjection();
    }

    // Project both stored levels to avoid creating mean velocity. LRC history is not yet
    // transformed consistently, so projection rejects frequency-dependent walls.
    void SetMeanProjection(int every) {
        if (every < 0) throw std::runtime_error("mean-projection interval cannot be negative");
        if (!every) {
            ProjectionEvery = ProjectionStep = 0;
            return;
        }
        if (!Pad || !InsideNodes) throw std::runtime_error("mean projection needs a walled room");
        if (P.Nw) throw std::runtime_error("mean projection does not yet carry LRC wall history");
        if (!ProjectReduce) {
            auto &ctx = MetalContext::Get();
            ProjectReduce = ctx.RoomPipeline("RoomImplicitReduce");
            ProjectSubtract = ctx.RoomPipeline("RoomImplicitProject");
            constexpr size_t values_per_group = 256 * 4;
            const size_t partials = (Nodes + values_per_group - 1) / values_per_group;
            for (auto &b : ProjectScratch) b.ResizeZeroed(std::max<size_t>(partials, 1) * sizeof(float));
        }
        ProjectPair();
        ProjectionEvery = every;
        ProjectionStep = 0;
    }

    const GpuBuffer &ReduceLevel(const GpuBuffer &level) {
        constexpr uint32_t threads = 256, values_per_thread = 4;
        auto &ctx = MetalContext::Get();
        const GpuBuffer *input = &level;
        size_t count = Nodes;
        int slot = 0;
        do {
            const uint32_t groups = uint32_t((count + size_t(threads * values_per_thread) - 1) / size_t(threads * values_per_thread));
            const RoomImplicitReduceParams rp{int(count)};
            GpuBuffer const *output = &ProjectScratch[slot];
            ctx.Dispatch(ProjectReduce, {groups, 1, 1}, {threads, 1, 1}, {input, output}, &rp, sizeof rp);
            input = output;
            slot ^= 1;
            count = groups;
        } while (count > 1);
        return *input;
    }

    double MeanLevel(const GpuBuffer &level) {
        return double(ReduceLevel(level).As<float>()[0]) / double(InsideNodes);
    }

    void ProjectLevel(GpuBuffer &level) {
        const GpuBuffer &sum = ReduceLevel(level);
        const RoomImplicitProjectParams pp{Gx, Gy, Gz, float(1. / double(InsideNodes))};
        MetalContext::Get().Dispatch(ProjectSubtract, Tiles, Threads, {&level, &sum, &Blocks}, &pp, sizeof pp);
    }

    void ProjectPair() {
        ProjectLevel(Cur);
        ProjectLevel(Prev);
    }

    void AdvanceMeanProjection() {
        if (ProjectionEvery && ++ProjectionStep == ProjectionEvery) {
            ProjectPair();
            ProjectionStep = 0;
        }
    }

    // Source and receiver signals are row-major by node.
    void SetIo(const std::vector<int> &in_nodes, const std::vector<float> &in_sigs, const std::vector<int> &out_nodes, int nt) {
        IoPso = MetalContext::Get().RoomPipeline("RoomImplicitIo");
        P.Ns = int(in_nodes.size());
        P.Nr = int(out_nodes.size());
        P.Nt = nt;
        InIxyz.ResizeZeroed(in_nodes.size() * sizeof(int));
        InIxyz.Upload(in_nodes.data(), in_nodes.size() * sizeof(int));
        InSigs.ResizeZeroed(size_t(P.Ns) * size_t(nt) * sizeof(float));
        InSigs.Upload(in_sigs.data(), in_sigs.size() * sizeof(float));
        OutIxyz.ResizeZeroed(out_nodes.size() * sizeof(int));
        OutIxyz.Upload(out_nodes.data(), out_nodes.size() * sizeof(int));
        Out.ResizeZeroed(size_t(P.Nr) * size_t(nt) * sizeof(float));
        PBuf.ResizeZeroed(sizeof P);
        PBuf.Upload(&P, sizeof P);
    }

    void StepIo(int n) {
        Step(true, false);
        const RoomStepParams st{n};
        const uint32_t io = uint32_t(P.Nr + P.Ns);
        MetalContext::Get().Dispatch(IoPso, {(io + 63) / 64, 1, 1}, {64, 1, 1}, {&Cur, &Out, &Prev, &OutIxyz, &InSigs, &InIxyz, &PBuf}, &st, sizeof st);
        AdvanceMeanProjection();
    }

    const ImplicitScheme &Scheme;
    double Lambda;
    ImplicitWall Wall;
    RoomMaterials Mats;
    bool Compensated;
    ImplicitShape Shape;
    int Pad;
    int Nx, Ny, Nz;
    int Gx, Gy, Gz;
    RoomImplicitParams P;
    size_t Nodes, Room;
    size_t InsideNodes{0};
    int WallNodes{0};
    int Codes{1};
    // Sparse G values for host energy accounting.
    std::vector<std::pair<int, double>> WallG;
    GpuBuffer Prev, Cur, Bp, Tmp, Blocks, Code, Terms, Keep;
    GpuBuffer WallList, MatMb, MatQuads, Vh, Gh, Ub;
    GpuBuffer InIxyz, InSigs, OutIxyz, Out, PBuf;
    GpuBuffer ProjectScratch[2];
    int ProjectionEvery{0}, ProjectionStep{0};
    MTL::ComputePipelineState *Rhs, *Jacobi, *Stream, *RhsFused{}, *JacobiFused{}, *WallRhsPso{}, *WallStepPso{}, *IoPso{}, *ProjectReduce{}, *ProjectSubtract{};
    Dim3 Tiles, Threads, WallTiles{}, WallThreads{};
};
// Mesh geometry supplies 26-leg crossings and nearest-triangle normals.
struct ImplicitMesh {
    ImplicitMesh(Eigen::MatrixXd v, Eigen::MatrixXi f) : V(std::move(v)), F(std::move(f)) {
        Tree.Init(V, F);
        Normals.resize(F.rows(), 3);
        for (int t = 0; t < F.rows(); ++t) {
            const Eigen::RowVector3d a = V.row(F(t, 0)), u = Eigen::RowVector3d(V.row(F(t, 1))) - a, v = Eigen::RowVector3d(V.row(F(t, 2))) - a;
            const Eigen::RowVector3d n{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
            Normals.row(t) = n.normalized();
        }
    }

    bool Blocked(int ix, int iy, int iz, int dx, int dy, int dz) const {
        const Eigen::RowVector3d a{double(ix), double(iy), double(iz)};
        double t = 0.;
        return Tree.Crossing(a, a + Eigen::RowVector3d{double(dx), double(dy), double(dz)}, t) >= 0;
    }

    void Normal(int ix, int iy, int iz, double n[3]) const {
        int t = -1;
        Eigen::RowVector3d closest;
        Tree.ClosestPoint(Eigen::RowVector3d{double(ix), double(iy), double(iz)}, t, closest);
        for (int a = 0; a < 3; ++a) n[a] = t < 0 ? 0. : Normals(t, a);
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd Normals;
    AabbTree<double> Tree;
};

double ImplicitMeshSample(const ImplicitMesh &mesh, int ix, int iy, int iz, const double nex[3], int k[3]) {
    double n[3];
    mesh.Normal(ix, iy, iz, n);
    k[0] = k[1] = k[2] = 0;
    double beta = 0.;
    for (const auto &leg : ImplicitLegs) {
        if (!mesh.Blocked(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz)) {
            ++k[leg.Class];
            continue;
        }
        beta += nex[leg.Class] * (n[0] * leg.Dx + n[1] * leg.Dy + n[2] * leg.Dz);
    }
    return beta;
}

// Outward-wound faces for corners encoded by the signs of x, y, and z.
Eigen::MatrixXi ImplicitBoxFaces() {
    static constexpr int Faces[6][4]{{0, 2, 6, 4}, {1, 5, 7, 3}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 1, 3, 2}, {4, 6, 7, 5}};
    Eigen::MatrixXi f(12, 3);
    for (int face = 0; face < 6; ++face) {
        f.row(2 * face) << Faces[face][0], Faces[face][2], Faces[face][1];
        f.row(2 * face + 1) << Faces[face][0], Faces[face][3], Faces[face][2];
    }
    return f;
}

ImplicitMesh ImplicitBoxMesh(const ImplicitShape &shape) {
    Eigen::MatrixXd v(8, 3);
    for (int corner = 0; corner < 8; ++corner) {
        const double q[3]{(corner & 1 ? 1. : -1.) * shape.Ext[0], (corner & 2 ? 1. : -1.) * shape.Ext[1], (corner & 4 ? 1. : -1.) * shape.Ext[2]};
        v(corner, 0) = shape.Cen[0] + shape.Cos * q[0] - shape.Sin * q[1];
        v(corner, 1) = shape.Cen[1] + shape.Sin * q[0] + shape.Cos * q[1];
        v(corner, 2) = shape.Cen[2] + q[2];
    }
    return {std::move(v), ImplicitBoxFaces()};
}

// PFFDTD export converted from meters to cell coordinates. Triangle winding is not trusted;
// each wall node orients its nearest normal against its ghost set.
struct ImplicitModel {
    std::vector<std::string> Materials;
    std::vector<int> Mat;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::RowVector3d Source;
    std::vector<Eigen::RowVector3d> Receivers;
    int N[3]{0, 0, 0};
};

ImplicitModel ImplicitLoadModel(const std::string &path, double h, int margin) {
    std::ifstream stream{path};
    if (!stream) throw std::runtime_error("Failed to read room model: " + path);
    nlohmann::json model;
    stream >> model;

    ImplicitModel out;
    std::vector<std::array<double, 3>> pts;
    std::vector<std::array<int, 3>> tris;
    for (const auto &entry : model["mats_hash"].items()) {
        const int base = int(pts.size()), mat = int(out.Materials.size());
        if (mat > std::numeric_limits<int8_t>::max()) throw std::runtime_error("Implicit room model has more than 128 materials");
        out.Materials.push_back(entry.key());
        for (const auto &p : entry.value()["pts"]) pts.push_back({p[0], p[1], p[2]});
        for (const auto &t : entry.value()["tris"]) {
            tris.push_back({base + int(t[0]), base + int(t[1]), base + int(t[2])});
            out.Mat.push_back(mat);
        }
    }
    if (tris.empty()) throw std::runtime_error("Room model carries no triangles: " + path);

    double lo[3]{1e30, 1e30, 1e30}, hi[3]{-1e30, -1e30, -1e30};
    for (const auto &p : pts)
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], p[a]);
            hi[a] = std::max(hi[a], p[a]);
        }
    out.V.resize(Eigen::Index(pts.size()), 3);
    for (size_t i = 0; i < pts.size(); ++i)
        for (int a = 0; a < 3; ++a) out.V(Eigen::Index(i), a) = (pts[i][size_t(a)] - lo[a]) / h + margin;
    out.F.resize(Eigen::Index(tris.size()), 3);
    for (size_t i = 0; i < tris.size(); ++i)
        for (int a = 0; a < 3; ++a) out.F(Eigen::Index(i), a) = tris[i][size_t(a)];
    for (int a = 0; a < 3; ++a) out.N[a] = int(std::ceil((hi[a] - lo[a]) / h)) + 2 * margin + 1;

    const auto &source = model["sources"][0]["xyz"];
    for (int a = 0; a < 3; ++a) out.Source[a] = (double(source[a]) - lo[a]) / h + margin;
    if (model.contains("receivers")) {
        for (const auto &receiver : model["receivers"]) {
            const auto &xyz = receiver.contains("xyz") ? receiver["xyz"] : receiver;
            Eigen::RowVector3d p;
            for (int a = 0; a < 3; ++a) p[a] = (double(xyz[a]) - lo[a]) / h + margin;
            out.Receivers.push_back(p);
        }
    }
    return out;
}

// Compares mesh crossings with an exact slab test for the same yawed box.
void ImplicitVoxelRung(const ImplicitScheme &scheme, const ImplicitShape &shape) {
    const double t0 = Now();
    const ImplicitMesh mesh = ImplicitBoxMesh(shape);
    const double nex[3]{ImplicitNex(scheme, 0), ImplicitNex(scheme, 1), ImplicitNex(scheme, 2)};

    auto local = [&](double x, double y, double z, double q[3]) {
        const double p[3]{x - shape.Cen[0], y - shape.Cen[1], z - shape.Cen[2]};
        shape.Local(p, q);
    };
    auto within = [&](const double q[3]) {
        return std::abs(q[0]) <= shape.Ext[0] && std::abs(q[1]) <= shape.Ext[1] && std::abs(q[2]) <= shape.Ext[2];
    };
    const auto crosses = [&](int ix, int iy, int iz, int dx, int dy, int dz) {
        double q0[3], q1[3];
        local(ix, iy, iz, q0);
        local(ix + dx, iy + dy, iz + dz, q1);
        if (within(q0) && within(q1)) return false;
        double lo = 0., hi = 1.;
        for (int a = 0; a < 3; ++a) {
            const double d = q1[a] - q0[a];
            if (std::abs(d) < 1e-15) {
                if (std::abs(q0[a]) > shape.Ext[a]) return false;
                continue;
            }
            double t0a = (-shape.Ext[a] - q0[a]) / d, t1a = (shape.Ext[a] - q0[a]) / d;
            if (t0a > t1a) std::swap(t0a, t1a);
            lo = std::max(lo, t0a);
            hi = std::min(hi, t1a);
            if (lo > hi) return false;
        }
        return true;
    };

    size_t legs = 0, wrong_leg = 0, nodes = 0, wrong_normal = 0, wrong_count = 0, inset = 0;
    double worst_beta = 0.;
    for (int ix = 0; ix < shape.N[0]; ++ix) {
        for (int iy = 0; iy < shape.N[1]; ++iy) {
            for (int iz = 0; iz < shape.N[2]; ++iz) {
                for (const auto &leg : ImplicitLegs) {
                    ++legs;
                    wrong_leg += size_t(mesh.Blocked(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz) != crosses(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz));
                }
                double na[3];
                shape.Normal(ix, iy, iz, na);
                int ka[3]{0, 0, 0}, km[3];
                double a = 0.;
                for (const auto &leg : ImplicitLegs) {
                    if (crosses(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz)) a += nex[leg.Class] * (na[0] * leg.Dx + na[1] * leg.Dy + na[2] * leg.Dz);
                    else ++ka[leg.Class];
                }
                double q[3];
                local(ix, iy, iz, q);
                inset += size_t(within(q) != shape.Inside(ix, iy, iz));
                if (ka[0] == 6 && ka[1] == 12 && ka[2] == 8) continue;
                ++nodes;
                const double m = ImplicitMeshSample(mesh, ix, iy, iz, nex, km);
                double nm[3];
                mesh.Normal(ix, iy, iz, nm);
                double dot = 0.;
                for (int c = 0; c < 3; ++c) dot += na[c] * nm[c];
                wrong_normal += size_t(dot < 1. - 1e-9);
                wrong_count += size_t(ka[0] != km[0] || ka[1] != km[1] || ka[2] != km[2]);
                worst_beta = std::max(worst_beta, std::abs(a - m));
            }
        }
    }
    std::printf("[implicit] %s voxelising a %.1f degree box mesh over a %d^3 grid, %d triangles\n", scheme.Name, std::atan2(shape.Sin, shape.Cos) * 180. / std::numbers::pi, shape.N[0], int(mesh.F.rows()));
    std::printf("  %zu legs against the exact segment test: %zu disagree\n", legs, wrong_leg);
    std::printf("  %zu wall nodes: %zu with kept counts off the closed form's, %zu with a normal off it, worst beta difference %.2e\n", nodes, wrong_count, wrong_normal, worst_beta);
    std::printf("  %zu nodes the half-cell inset keeps and the crossing rule does not, or the other way | %.1fs\n", inset, Now() - t0);
}

// Voxelizes all legs, fills from the source, and assigns nearest normals and materials.
ImplicitVoxels ImplicitVoxelise(const ImplicitModel &model, const ImplicitMesh &mesh) {
    ImplicitVoxels vox;
    std::copy_n(model.N, 3, vox.N);
    const int nx = vox.N[0], ny = vox.N[1], nz = vox.N[2];
    const size_t nodes = size_t(nx) * size_t(ny) * size_t(nz);
    vox.Blocked.assign(nodes, 0);
    vox.Filled.assign(nodes, 0);
    vox.Normals.assign(3 * nodes, 0.f);
    vox.Mat.assign(nodes, -1);

    // Relaxed atomics publish symmetric endpoint hits without a second per-node mask. The
    // parallel-region barrier completes every mask before the fill begins.
    const auto mark_blocked = [&](size_t i, int bit) {
        std::atomic_ref<uint32_t>(vox.Blocked[i]).fetch_or(1u << bit, std::memory_order_relaxed);
    };
    ParallelFor(size_t(nx), 1, [&](size_t sx) {
        const int ix = int(sx);
        for (int iy = 0; iy < ny; ++iy)
            for (int iz = 0; iz < nz; ++iz) {
                const size_t i = vox.At(ix, iy, iz);
                for (const auto &leg : ImplicitLegs) {
                    const int jx = ix + leg.Dx, jy = iy + leg.Dy, jz = iz + leg.Dz;
                    if (jx < 0 || jy < 0 || jz < 0 || jx >= nx || jy >= ny || jz >= nz) {
                        if (mesh.Blocked(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz)) mark_blocked(i, leg.Bit);
                        continue;
                    }
                    if (leg.Bit > leg.OppositeBit) continue;
                    if (!mesh.Blocked(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz) && !mesh.Blocked(jx, jy, jz, -leg.Dx, -leg.Dy, -leg.Dz)) continue;
                    mark_blocked(i, leg.Bit);
                    mark_blocked(vox.At(jx, jy, jz), leg.OppositeBit);
                }
            }
    });

    // Fill over the operator's full 26-leg graph; a six-leg fill would create unmatched
    // diagonal Dirichlet edges and break symmetry.
    std::vector<size_t> queue;
    const int seed[3]{int(std::lround(model.Source[0])), int(std::lround(model.Source[1])), int(std::lround(model.Source[2]))};
    queue.push_back(vox.At(seed[0], seed[1], seed[2]));
    vox.Filled[queue.front()] = 1;
    for (size_t head = 0; head < queue.size(); ++head) {
        const size_t i = queue[head];
        const int iz = int(i % size_t(nz)), iy = int((i / size_t(nz)) % size_t(ny)), ix = int(i / (size_t(nz) * size_t(ny)));
        for (const auto &leg : ImplicitLegs) {
            const int jx = ix + leg.Dx, jy = iy + leg.Dy, jz = iz + leg.Dz;
            if (jx < 0 || jy < 0 || jz < 0 || jx >= nx || jy >= ny || jz >= nz) continue;
            if (vox.Blocked[i] & (1u << leg.Bit)) continue;
            const size_t j = vox.At(jx, jy, jz);
            if (vox.Filled[j]) continue;
            vox.Filled[j] = 1;
            queue.push_back(j);
        }
    }

    ParallelFor(size_t(nx), 1, [&](size_t sx) {
        const int ix = int(sx);
        for (int iy = 0; iy < ny; ++iy)
            for (int iz = 0; iz < nz; ++iz) {
                const size_t i = vox.At(ix, iy, iz);
                if (!vox.Filled[i] || !vox.Blocked[i]) continue;
                int tri = -1;
                Eigen::RowVector3d closest;
                mesh.Tree.ClosestPoint(Eigen::RowVector3d{double(ix), double(iy), double(iz)}, tri, closest);
                if (tri < 0) continue;
                double n[3], stair[3]{0., 0., 0.};
                for (int a = 0; a < 3; ++a) n[a] = mesh.Normals(tri, a);
                for (const auto &leg : ImplicitLegs) {
                    if (!(vox.Blocked[i] & (1u << leg.Bit))) continue;
                    const double len = std::sqrt(double(leg.Class + 1));
                    stair[0] += leg.Dx / len;
                    stair[1] += leg.Dy / len;
                    stair[2] += leg.Dz / len;
                }
                const double align = n[0] * stair[0] + n[1] * stair[1] + n[2] * stair[2];
                for (int a = 0; a < 3; ++a) vox.Normals[3 * i + size_t(a)] = float(align < 0. ? -n[a] : n[a]);
                vox.Mat[i] = int8_t(model.Mat[size_t(tri)]);
            }
    });
    return vox;
}
