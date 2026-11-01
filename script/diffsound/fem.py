"""Linear/quadratic tetrahedral elasticity and generalized eigenvalue derivatives."""

import math
from pathlib import Path

import meshio
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
from torch import nn

from .dsp import Weighted

MATERIALS = {
    "Ceramic": (2700, 7.2e10, 0.19, 6, 1e-7),
    "Glass": (2600, 6.2e10, 0.20, 1, 1e-7),
    "Wood": (750, 1.1e10, 0.25, 60, 2e-6),
    "Plastic": (1070, 1.4e9, 0.35, 30, 1e-6),
    "Iron": (8000, 2.1e11, 0.28, 10, 1e-7),
    "Polycarbonate": (1190, 2.4e9, 0.37, 0.5, 4e-7),
    "Steel": (7850, 2e11, 0.29, 20, 3e-8),
    "Tin": (7265, 5e10, 0.325, 2, 3e-8),
}
EDGES = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def read_mesh(path):
    path = Path(path)
    if path.suffix == ".obj":
        path = Path(str(path) + "_.msh")
    mesh = meshio.read(path)
    return compact(
        torch.tensor(mesh.points, dtype=torch.float64), torch.tensor(mesh.cells_dict["tetra"], dtype=torch.long)
    )


def compact(vertices, tets):
    used, inverse = torch.unique(tets.reshape(-1), return_inverse=True)
    return vertices[used], inverse.reshape(tets.shape)


def quadratic(vertices, tets):
    edges, inverse = torch.unique(tets[:, EDGES].reshape(-1, 2).sort(dim=1).values, dim=0, return_inverse=True)
    mid = inverse.reshape(-1, 6) + len(vertices)
    nodes = torch.stack(
        (
            tets[:, 0],
            mid[:, 0],
            tets[:, 1],
            mid[:, 3],
            tets[:, 2],
            mid[:, 1],
            mid[:, 2],
            mid[:, 4],
            mid[:, 5],
            tets[:, 3],
        ),
        dim=1,
    )
    return torch.cat((vertices, vertices[edges].mean(1))), nodes


def quadrature(order):
    roots, weights = np.polynomial.legendre.leggauss(order + 2)
    roots = (roots + 1) / 2
    points, result = [], []
    for i, w in enumerate(roots):
        for j, r in enumerate(roots):
            z = r * (1 - w)
            for k, s in enumerate(roots):
                y = s * (1 - w - z)
                points.append([1 - w - z - y, y, z, w])
                result.append(weights[i] * weights[j] * weights[k] * (1 - w) * (1 - w - z) / 8)
    # Match the release's quadrature constants before accumulating in float64.
    return torch.tensor(np.array(points, np.float32), dtype=torch.float64), torch.tensor(
        np.array(result, np.float32), dtype=torch.float64
    )


def shape_functions(points, order):
    identity = torch.eye(4, dtype=points.dtype)
    if order == 1:
        return points, identity.expand(len(points), -1, -1)
    if order != 2:
        raise ValueError("The released mesh/mass assembly supports orders 1 and 2")
    values, grads = [], []
    for i, j in ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)):
        if i == j:
            values.append(points[:, i] * (2 * points[:, i] - 1))
            grads.append((4 * points[:, i] - 1)[:, None] * identity[i])
        else:
            values.append(4 * points[:, i] * points[:, j])
            grads.append(4 * (points[:, j, None] * identity[i] + points[:, i, None] * identity[j]))
    return torch.stack(values, 1), torch.stack(grads, 1)


class Material(nn.Module):
    def __init__(self, values, learn_poisson=True):
        super().__init__()
        self.values = tuple(values)
        self.youngs = Weighted(
            torch.exp(torch.linspace(math.log(values[1] / 10), math.log(values[1] * 10), 16)), device="cpu"
        )
        self.poisson = Weighted(torch.linspace(0.01, 0.499, 16) if learn_poisson else [values[2]], device="cpu")
        self.poisson.raw.requires_grad_(learn_poisson)

    def initialize(self, steps=5000):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = (self.youngs() / self.values[1] - 1).square() + (self.poisson() / self.values[2] - 1).square()
            loss.backward()
            optimizer.step()


class Elasticity:
    def __init__(self, vertices, tets, order=1):
        vertices, tets = compact(vertices.to("cpu", torch.float64), tets.cpu())
        if order == 2:
            vertices, tets = quadratic(vertices, tets)
        self.vertices, self.tets, self.order = vertices, tets, order
        corners = tets[:, [0, 1, 2, 3] if order == 1 else [0, 2, 4, 9]]
        p = vertices[corners]
        transform = (p[:, :3] - p[:, 3:]).transpose(1, 2)
        determinant = torch.linalg.det(transform).abs()
        if not torch.isfinite(determinant).all() or (determinant <= 0).any():
            raise ValueError("Non-finite or zero-volume tetrahedron")
        points, weights = quadrature(order)
        shape, derivative = shape_functions(points, order)
        derivative = derivative[..., :3] - derivative[..., 3:]
        gradient = derivative[None] @ torch.linalg.inv(transform)[:, None]
        gram = torch.einsum("eqia,eqjb,q,e->eiajb", gradient, gradient, weights, determinant)
        trace = torch.einsum("eiaja->eij", gram)
        mu = (
            gram.permute(0, 1, 4, 3, 2)
            + trace[:, :, None, :, None] * torch.eye(3, dtype=torch.float64)[None, None, :, None, :]
        )
        n = tets.shape[1] * 3
        self.kl = gram.reshape(-1, n, n)
        self.km = mu.reshape(-1, n, n)
        scalar_mass = torch.einsum("qi,qj,q->ij", shape, shape, weights)
        self.mass = (
            determinant[:, None, None, None, None]
            * scalar_mass[None, :, None, :, None]
            * torch.eye(3, dtype=torch.float64)[None, None, :, None, :]
        ).reshape(-1, n, n)
        self.dofs = (tets[..., None] * 3 + torch.arange(3)).reshape(-1, n)
        self.size = len(vertices) * 3

    def stiffness(self, youngs, poisson):
        lam = youngs * poisson / ((1 + poisson) * (1 - 2 * poisson))
        mu = youngs / (2 * (1 + poisson))
        return lam * self.kl + mu * self.km

    def sparse(self, elements):
        n = self.dofs.shape[1]
        indices = self.dofs.numpy()
        return sp.coo_matrix(
            (
                elements.detach().numpy().reshape(-1),
                (
                    np.broadcast_to(indices[:, :, None], (len(indices), n, n)).reshape(-1),
                    np.broadcast_to(indices[:, None, :], (len(indices), n, n)).reshape(-1),
                ),
            ),
            shape=(self.size, self.size),
        ).tocsr()

    def solve(self, youngs, poisson, density, modes):
        stiffness = self.sparse(self.stiffness(youngs, poisson))
        mass = self.sparse(self.mass * density)
        if modes + 6 >= self.size:
            raise ValueError("Too few mesh degrees of freedom for requested elastic modes")
        values, vectors = eigsh(
            stiffness, M=mass, k=modes + 6, sigma=20000, v0=np.random.default_rng(0).normal(size=self.size)
        )
        values, vectors = values[6:], vectors[:, 6:]
        residual = stiffness @ vectors - (mass @ vectors) * values
        error = np.linalg.norm(residual, axis=0) / (
            np.linalg.norm(stiffness @ vectors, axis=0) + np.linalg.norm((mass @ vectors) * values, axis=0)
        )
        if (values <= 0).any() or not np.isfinite(values).all() or error.max() > 1e-5:
            raise RuntimeError(f"Invalid elastic eigenpairs: relative residual {error.max():.3g}")
        return torch.from_numpy(values), torch.from_numpy(vectors)

    def values(self, youngs, poisson, density, eigenpairs):
        values, vectors = eigenpairs
        u = vectors[self.dofs]
        stiffness = self.stiffness(youngs, poisson)
        k = torch.einsum("eim,eij,ejm->m", u, stiffness, u)
        m = torch.einsum("eim,eij,ejm->m", u, self.mass * density, u)
        # Preserve upstream's frozen-eigenvector Rayleigh correction between refreshes.
        return values + k - values * m

    def projected(self, eigenpairs):
        """Cache material-independent modal energies for fixed-geometry fitting."""
        values, vectors = eigenpairs
        u = vectors[self.dofs]
        return tuple(torch.einsum("eim,eij,ejm->m", u, a, u).detach() for a in (self.kl, self.km, self.mass))
