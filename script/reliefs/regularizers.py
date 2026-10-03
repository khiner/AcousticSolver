"""Texture losses from Acoustic Reliefs; see ../../src/Bem/LICENSE.AcousticReliefs."""

import torch


def smoothness(tex_torch):
    data_padded = torch.nn.functional.pad(tex_torch, (1, 1, 1, 1), value=0)
    lap = (
        data_padded[:-2, 1:-1]
        + data_padded[2:, 1:-1]
        + data_padded[1:-1, :-2]
        + data_padded[1:-1, 2:]
        - 4 * data_padded[1:-1, 1:-1]
    )
    loss = torch.mean(lap * lap)
    return loss


def neg_relu(tex_torch):
    loss = torch.mean(torch.nn.functional.relu(-tex_torch))
    return loss


def barrier_loss(tex_torch, vmax):
    vl = vmax + 0.001
    loss = -torch.mean(
        torch.minimum(torch.log(tex_torch + vl), torch.log(vl - tex_torch))
    )
    return loss


def preprocess_tex(tex_torch, edge_border):
    hfield = torch.nn.functional.pad(
        tex_torch, (edge_border, edge_border, edge_border, edge_border), value=0
    )
    return hfield


def normalize_gradients(grad):
    norm = torch.linalg.norm(grad)
    if norm == 0:
        return grad
    return grad / norm
