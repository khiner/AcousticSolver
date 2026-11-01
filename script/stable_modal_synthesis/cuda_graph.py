"""Capture unchanged author SAV steps for long, inference-only CUDA trajectories."""

import torch


def forward(solver, ivp, chunk=32):
    current, parameters = solver._method.init(ivp)
    initial = current.clone()
    state = current.clone()
    force = torch.zeros((current.shape[0], chunk), dtype=current.dtype, device=current.device)
    parameters = parameters._replace(fe_points=force)
    recorded = torch.empty((*current.shape, chunk), dtype=current.dtype, device=current.device)
    output = torch.empty((*current.shape, ivp.num_samples), dtype=current.dtype, device=current.device)
    output[..., 0] = initial

    def steps():
        value = state
        for n in range(chunk):
            value = solver._method.step(n, value, parameters)
            recorded[..., n].copy_(value)
        state.copy_(value)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        steps()
    torch.cuda.current_stream().wait_stream(stream)
    state.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        steps()
    state.copy_(initial)
    for offset in range(0, ivp.num_samples - 1, chunk):
        count = min(chunk, ivp.num_samples - 1 - offset)
        force.zero_()
        force[:, :count].copy_(ivp.fe_points[:, offset : offset + count])
        graph.replay()
        output[..., offset + 1 : offset + 1 + count].copy_(recorded[..., :count])
    return output[..., -1] if solver._use_last else output


def install():
    from src.ode.adjoint import AutoDiffAdjoint

    original = AutoDiffAdjoint.forward

    def captured(self, ivp):
        if ivp.y0.is_cuda and not torch.is_grad_enabled() and ivp.num_samples >= 1000:
            return forward(self, ivp)
        return original(self, ivp)

    AutoDiffAdjoint.forward = captured
    return original
