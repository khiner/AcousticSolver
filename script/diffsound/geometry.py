"""Differentiable volumetric clipping, shell/morph parameters, and neural SDF."""

from pathlib import Path

import meshio
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from . import DEVICE
from .fem import EDGES, compact
from .tables import TET_TABLE, TRIANGLE_TABLE


def clip(vertices, sdf, tets, thickness=None):
    inside = sdf > 0
    if thickness is not None:
        inside = inside & (sdf <= thickness)
    occupancy = inside[tets]
    count = occupancy.sum(1)
    boundary = (count > 0) & (count < 4)
    ids = (occupancy[boundary] * torch.tensor([1, 2, 4, 8])).sum(1)
    edges, inverse = torch.unique(tets[boundary][:, EDGES].reshape(-1, 2).sort(1).values, dim=0, return_inverse=True)
    crossing = inside[edges].sum(1) == 1
    mapping = torch.full((len(edges),), -1, dtype=torch.long)
    mapping[crossing] = torch.arange(int(crossing.sum()))
    index = mapping[inverse].reshape(-1, 6)
    edges = edges[crossing]
    distance = sdf[edges]
    if thickness is not None:
        distance = distance - ((distance > 0).all(1))[:, None] * thickness
    weights = torch.stack((-distance[:, 1], distance[:, 0]), 1) / (distance[:, 0] - distance[:, 1])[:, None]
    surface = (vertices[edges] * weights[..., None]).sum(1)
    triangle_table, tet_table = torch.tensor(TRIANGLE_TABLE), torch.tensor(TET_TABLE)
    triangles, cells = [], []
    local = torch.cat((tets[boundary], index + len(vertices)), 1)
    for number in (1, 2):
        mask = (triangle_table[ids] >= 0).sum(1) == 3 * number
        triangles.append(index[mask].gather(1, triangle_table[ids[mask], : 3 * number]).reshape(-1, 3))
    for number in (1, 3):
        mask = (tet_table[ids] >= 0).sum(1) == 4 * number
        cells.append(local[mask].gather(1, tet_table[ids[mask], : 4 * number]).reshape(-1, 4))
    cells.append(tets[count == 4])
    points, cells = compact(torch.cat((vertices, surface)), torch.cat(cells))
    return points, cells, surface, torch.cat(triangles)


def surface(vertices, tets):
    faces = tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3).numpy()
    opposite = tets.reshape(-1).numpy()
    _, first, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    first = first[count == 1]
    faces, opposite = faces[first], opposite[first]
    points = vertices.detach().numpy()
    p = points[faces]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = (normal * (points[opposite] - p[:, 0])).sum(1) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    normal[inward] *= -1
    area = np.linalg.norm(normal, axis=1)
    normal /= area[:, None]
    return faces, normal, area


def largest_component(vertices, tets):
    edges = tets[:, EDGES].reshape(-1, 2).numpy()
    graph = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices))
    ).tocsr()
    _, labels = sp.csgraph.connected_components(graph, directed=False)
    keep = torch.from_numpy(labels == np.bincount(labels).argmax())
    return compact(vertices, tets[keep[tets].all(1)])


def load_grid(root, resolution, scale=1.0):
    with np.load(Path(root) / f"tets/{resolution}_tets.npz") as archive:
        return torch.tensor(archive["vertices"], dtype=torch.float32) * scale, torch.tensor(
            archive["indices"], dtype=torch.long
        )


def signed_distance(path, points):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles():
        raise ValueError(f"No triangle surface in {path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return -torch.from_numpy(scene.compute_signed_distance(points.detach().numpy().astype(np.float32)).numpy())


def save_mesh(path, vertices, cells):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = "triangle" if cells.shape[1] == 3 else "tetra"
    meshio.write(
        path,
        meshio.Mesh(vertices.detach().cpu().numpy(), [(kind, cells.cpu().numpy())]),
        **({"file_format": "gmsh22", "binary": False} if path.suffix == ".msh" else {}),
    )


class NeuralSDF(nn.Module):
    def __init__(self, vertices, tets, resolution, scale, frequencies=3):
        super().__init__()
        self.resolution, self.scale, self.frequencies = resolution, scale, frequencies
        self.register_buffer("vertices", vertices.to(DEVICE))
        self.register_buffer("tets", tets)
        self.deform = nn.Parameter(torch.zeros_like(self.vertices))
        layers = [nn.Linear(6 * frequencies + 3, 512, device=DEVICE), nn.ReLU()]
        for _ in range(3):
            layers.extend((nn.Linear(512, 512, device=DEVICE), nn.ReLU()))
        layers.append(nn.Linear(512, 1, device=DEVICE))
        self.network = nn.Sequential(*layers)

    def sdf(self, points):
        parts = [points]
        for i in range(self.frequencies):
            phase = 2**i * torch.pi * points / self.scale
            parts.extend((phase.sin(), phase.cos()))
        return self.network(torch.cat(parts, -1)).flatten()

    def template_loss(self, points, distances, margin=0.0):
        sdf = self.sdf(points)
        wrong_inside = (distances > margin) & (sdf <= margin)
        wrong_outside = (distances < -margin) & (sdf >= margin)
        if not (wrong_inside.any() or wrong_outside.any()):
            return None
        return (-sdf[wrong_inside].sum() + sdf[wrong_outside].sum()) / self.resolution**3 * 1000

    def forward(self):
        points = self.vertices + self.scale * 1.8 / (self.resolution * 2) * self.deform.tanh()
        return clip(points.cpu(), self.sdf(points / self.scale).cpu(), self.tets)


def voxel_samples(path, resolution):
    mesh = meshio.read(path)
    low, high = mesh.points.min(0), mesh.points.max(0)
    center = (low + high) / 2
    size = (high - low).max() * 1.05
    low, high = center - size / 2, center + size / 2
    ranges = np.linspace(low, high, resolution)
    points = torch.tensor(np.stack(np.meshgrid(*ranges.T), axis=-1).astype(np.float32)).reshape(-1, 3)
    distances = signed_distance(path, points)
    return (
        ((points - torch.tensor(low, dtype=torch.float32)) / size - 0.5).to(DEVICE),
        distances.to(DEVICE),
        float(size),
    )
