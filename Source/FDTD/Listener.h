#pragma once

// Ported from WaveBlender (c) 2024 Kangrui Xue (GPUListener.h) — Metal port.

#include <fstream>
#include <string>

#include "KernelParams.h"
#include "MetalContext.h"

// Base listener class for writing FDTDSolver output.
class Listener {
public:
    // Adds a single pressure field sample. `s` is the sample index within the current batch.
    virtual void AddSample(const GpuBuffer &p, int s) = 0;

    // Writes a batch of pressure samples to file.
    virtual void Write() = 0;

    virtual ~Listener() = default;

protected:
    GpuBuffer Buf; // output buffer (device)
    int BufSize{0}; // number of samples in buffer

    std::ofstream OutFile;
};

// Mono-channel, stationary listener for single-point pressure output.
class MonoListener : public Listener {
public:
    MonoListener(int listener_cid, int n_fdtd_samples, const std::string &output_name = "output") : ListenerCid(listener_cid) {
        BufSize = n_fdtd_samples;
        Buf.ResizeZeroed(n_fdtd_samples * sizeof(float));
        OutFile.open(output_name + ".bin", std::ofstream::binary);
    }

    // Samples pressure at the listening position into Buf[s], on the GPU timeline.
    void AddSample(const GpuBuffer &p, int s) override {
        const SampleListenerParams params{ListenerCid, s};
        MetalContext::Get().Dispatch("sample_listener", {1, 1, 1}, {1, 1, 1}, {&p, &Buf}, &params, sizeof(params));
    }

    void Write() override {
        const auto *out = Buf.As<float>(); // synchronizes the stream
        OutFile.write(reinterpret_cast<const char *>(out), BufSize * sizeof(float));
    }

private:
    int ListenerCid{0};
};
