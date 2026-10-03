"""Small generated-mesh check of the Metal solve and texture derivative."""

import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sphere():
    vertices = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    faces = []
    for i in (0, 1):
        for j in (2, 3):
            for k in (4, 5):
                face = [i, j, k]
                if np.dot(np.cross(np.subtract(vertices[j], vertices[i]), np.subtract(vertices[k], vertices[i])), vertices[i]) < 0:
                    face = [i, k, j]
                faces.append(face)
    edges = {}
    refined = []

    def midpoint(a, b):
        key = tuple(sorted((a, b)))
        if key not in edges:
            point = np.add(vertices[a], vertices[b]).astype(float)
            point /= np.linalg.norm(point)
            edges[key] = len(vertices)
            vertices.append(point.tolist())
        return edges[key]

    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        refined.extend(([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]))
    return (np.asarray(vertices) * 0.25).tolist(), refined


def run(executable, payload, output, reference=False):
    source = output.with_suffix(".input.json")
    source.write_text(json.dumps(payload) + "\n")
    command = [str(ROOT / "build" / executable), str(source), str(output)]
    if reference:
        command.append("--reference")
    subprocess.run(command, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return json.loads(output.read_text())


def main():
    vertices, triangles = sphere()
    problem = dict(
        vertices=vertices,
        triangles=triangles,
        source=[0, 2, 0],
        listeners=[[1, 1, 0], [-1, 1, 0], [0, 1, 1], [0, 1, -1], [0, 2, 1]],
        height_vertices=[i for i, point in enumerate(vertices) if point[1] > 0],
        frequency_hz=500,
    )
    with tempfile.TemporaryDirectory(prefix="bem-sanity-") as folder:
        path = Path(folder)
        metal = run("BemSolve", problem, path / "metal.json")
        cpu = run("BemSolve", problem, path / "cpu.json", reference=True)
        pressure = np.asarray(metal["scattered_pressure"]) - np.asarray(cpu["scattered_pressure"])
        pressure_error = np.linalg.norm(pressure) / np.linalg.norm(cpu["scattered_pressure"])
        gradient = np.asarray(metal["height_gradient"]) - np.asarray(cpu["height_gradient"])
        gradient_error = np.linalg.norm(gradient) / np.linalg.norm(cpu["height_gradient"])
        assert pressure_error < 1e-4, pressure_error
        assert gradient_error < 1e-3, gradient_error
        assert metal["independent_forward_residual"] < 1e-4
        assert metal["independent_adjoint_residual"] < 1e-4

        pixels = np.zeros((8, 8), dtype=np.float32)
        def evaluate(value, name):
            pixels[4, 4] = value
            design = dict(problem=problem, design=dict(width=8, height=8, initial_pixels=pixels.ravel().tolist()))
            return run("BemTextureSolve", design, path / f"{name}.json")

        center = evaluate(0, "center")
        step = 0.001
        plus = evaluate(step, "plus")
        minus = evaluate(-step, "minus")
        finite_difference = (plus["loss"] - minus["loss"]) / (2 * step)
        derivative = center["gradient"][4 * 8 + 4]
        derivative_error = abs(derivative - finite_difference) / max(abs(finite_difference), 1e-8)
        assert derivative_error < 0.02, derivative_error
        print(json.dumps(dict(pressure_error=pressure_error, height_gradient_error=gradient_error, texture_derivative_error=derivative_error)))


if __name__ == "__main__":
    main()
