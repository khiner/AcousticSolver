// Ported from WaveBlender (c) 2024 Kangrui Xue (GPUSolver.cu) — Metal port.
//
// References:
//   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains.

#include "MetalSolver.h"

// All timesteps of a batch are encoded into one command buffer and flushed when a
// listener (or the next batch's CPU phase) first touches a shared buffer.
void FDTDSolver::RunFdtd() {
    auto &ctx = MetalContext::Get();

    const Dim3 fdtd_threads{8, 8, 8};
    const Dim3 fdtd_blocks{uint32_t(Params.Nx + 7) / 8, uint32_t(Params.Ny + 7) / 8, uint32_t(Params.Nz + 7) / 8};

    const Dim3 shader_threads{32, 1, 1};
    const Dim3 shader_blocks{uint32_t(NShaderPoints + 31) / 32, 1, 1};

    for (int s = 0; s < NFdtdSamples; ++s) {
        REAL t = (s + 1 == NFdtdSamples) ? 1. : REAL(s + 1) / NFdtdSamples; // normalized blending time (0, 1]
        const REAL ss = t * (NShaderSamples - 1); // shader sample index (fractional to support interpolation)
        if (Params.Scheme == BlendScheme::NoBlend) t = (s + 1 == NFdtdSamples) ? 1. : 0.;

        const StepPressureParams p_params{t, RhoCCDt, InvDx, Params.Nx, Params.Ny, Params.Nz};
        ctx.Dispatch("step_pressure", fdtd_blocks, fdtd_threads, {&P, &Px, &Py, &Pz, &Vx, &Vy, &Vz, &Beta, &Cell, &PmlNp, &PmlDp}, &p_params, sizeof(p_params));

        const ApplyShaderParams s_params{NShaderSamples, NShaderPoints, InvRhoDt, Params.Nx, Params.Ny, Params.Nz, ss};
        ctx.Dispatch("apply_shader", shader_blocks, shader_threads, {&Vx, &Vy, &Vz, &Beta, &ShaderData, &ShaderMap}, &s_params, sizeof(s_params));

        const StepVelocityParams v_params{InvRhoDt, InvDx, Params.Nx, Params.Ny, Params.Nz};
        ctx.Dispatch("step_velocity", fdtd_blocks, fdtd_threads, {&Vx, &Vy, &Vz, &P, &Beta, &PmlNv, &PmlDv}, &v_params, sizeof(v_params));

        for (auto *listener : Listeners) listener->AddSample(P, s);
        ++Step;
    }
    for (auto *listener : Listeners) listener->Write();
}

FDTDSolver::FDTDSolver(const SimParams &params) : Params(params), GridSize(Params.Nx * Params.Ny * Params.Nz), NFdtdSamples(Params.FdtdSrate / Params.BlendRate), NShaderSamples(Params.ShaderSrate / Params.BlendRate + 1), RhoCCDt(Params.Rho * Params.C * Params.C * Params.Dt), InvDx(1. / Params.Dx), InvRhoDt(1. / Params.Rho * Params.Dt), Damping(Params.Damping) {
    P.ResizeZeroed(GridSize * sizeof(REAL));
    Vx.ResizeZeroed(GridSize * sizeof(REAL));
    Vy.ResizeZeroed(GridSize * sizeof(REAL));
    Vz.ResizeZeroed(GridSize * sizeof(REAL));

    Beta.ResizeZeroed(GridSize * sizeof(REAL));
    Cell.ResizeZeroed(GridSize * sizeof(int));

    InitializePml();
}

void FDTDSolver::AddListener(const std::string &format, const std::vector<REAL> &position, const std::string &output_name) {
    const int i = (position[0] / Params.Dx) + (Params.Nx - 1) / 2.;
    const int j = (position[1] / Params.Dx) + (Params.Ny - 1) / 2.;
    const int k = (position[2] / Params.Dx) + (Params.Nz - 1) / 2.;

    if (format != "Mono") throw std::runtime_error(std::format("Invalid listener format: {}", format));
    Listeners.push_back(new MonoListener{Cid(i, j, k), NFdtdSamples, output_name});
}

void FDTDSolver::LogZSlice(const std::string &filetag) {
    const int offset = Cid(0, 0, Params.Nz / 2); // z-slice
    std::ofstream logfile{filetag, std::ofstream::binary};

    const auto *p = P.As<REAL>(); // synchronizes the stream
    logfile.write(reinterpret_cast<const char *>(p + offset), Params.Nx * Params.Ny * sizeof(REAL));

    const auto *beta = Beta.As<REAL>();
    logfile.write(reinterpret_cast<const char *>(beta + offset), Params.Nx * Params.Ny * sizeof(REAL));
}

// Same simple quadratic-ramp split-field PML as the reference (see its TODO on C-PML).
void FDTDSolver::InitializePml() {
    Px.ResizeZeroed(GridSize * sizeof(REAL));
    Py.ResizeZeroed(GridSize * sizeof(REAL));
    Pz.ResizeZeroed(GridSize * sizeof(REAL));

    const int max_half_grid_length = (std::max({Params.Nx, Params.Ny, Params.Nz}) + 1) / 2;

    // Velocity and pressure PML weights: numerator and denominator
    std::vector<REAL> pml_nv(max_half_grid_length + 1, 1.), pml_dv(max_half_grid_length + 1, 1. / (1. + Damping));
    std::vector<REAL> pml_np(max_half_grid_length, 1.), pml_dp(max_half_grid_length, 1.);

    for (int dist = 0; dist < PML_WIDTH; ++dist) {
        REAL weight = (REAL(PML_WIDTH) - dist) / PML_WIDTH; // velocity
        weight = 0.5 * weight * weight;
        pml_nv[dist] = 1. - weight;
        pml_dv[dist] = 1. / (1. + weight);

        weight = (REAL(PML_WIDTH) - dist - 0.5) / PML_WIDTH; // pressure
        weight = 0.5 * weight * weight;
        pml_np[dist] = 1. - weight;
        pml_dp[dist] = 1. / (1. + weight);
    }
    PmlNv.Resize((max_half_grid_length + 1) * sizeof(REAL));
    PmlDv.Resize((max_half_grid_length + 1) * sizeof(REAL));
    PmlNv.Upload(pml_nv.data(), (max_half_grid_length + 1) * sizeof(REAL));
    PmlDv.Upload(pml_dv.data(), (max_half_grid_length + 1) * sizeof(REAL));

    PmlNp.Resize(max_half_grid_length * sizeof(REAL));
    PmlDp.Resize(max_half_grid_length * sizeof(REAL));
    PmlNp.Upload(pml_np.data(), max_half_grid_length * sizeof(REAL));
    PmlDp.Upload(pml_dp.data(), max_half_grid_length * sizeof(REAL));
}

FDTDSolver::~FDTDSolver() {
    for (const auto *listener : Listeners) delete listener;
}
