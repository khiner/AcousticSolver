// Ported from WaveBlender (c) 2024 Kangrui Xue (SpeakerShader.cu) — Metal port.
// Implements the Speaker acoustic shader for pre-recorded input audio (kernel: speaker in Kernels.metal).
//
// References:
//   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains

#include "Shaders.h"

#include "AudioFile.h"
#include "KernelParams.h"

void Speaker::ReadWav(const std::string &wav_file) {
    AudioFile<REAL> audio_file;
    audio_file.load(wav_file);
    if (audio_file.getSampleRate() != Obj.Srate) throw std::runtime_error("Mismatch in input audio sample rate!");

    std::vector<REAL> audio_vn;
    REAL an1 = 0., an2 = 0.;
    for (int i = 0; i < audio_file.getNumSamplesPerChannel(); ++i) {
        an1 = an2;
        an2 = audio_file.samples[0][i]; // read next sample (assumes mono channel)

        const REAL vn = !audio_vn.empty() ? audio_vn.back() : 0.f;
        audio_vn.push_back(vn + (an1 + an2) / 2. * Obj.Dt); // Trapezoidal rule (bilinear transform)
    }
    Audio.Resize(audio_vn.size() * sizeof(REAL));
    Audio.Upload(audio_vn.data(), audio_vn.size() * sizeof(REAL));
}

void Speaker::Compute(GpuBuffer &vb, int global_bid) {
    const Dim3 threads{16, 16, 1};
    const Dim3 blocks{uint32_t(Obj.NPoints + 15) / 16, uint32_t(Obj.NSamples + 15) / 16, 1};

    const SpeakerParams params{global_bid, Direction, Obj.NPoints, Obj.NSamples, Obj.Step};
    MetalContext::Get().Dispatch("speaker", blocks, threads, {&vb, &Obj.GpuBN, &Audio}, &params, sizeof(params));
    Obj.Step += Obj.NSamples - 1;

    if (Obj.AnimFile.is_open()) Obj.ReadAnimation();
    else if (Obj.Step > 0) Obj.Changed = false;
}
