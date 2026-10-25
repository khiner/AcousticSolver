"""NumCalc scene preparation, provenance, and complex field output."""
import hashlib
import json
from itertools import groupby
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'external/mesh2hrtf'
PAPER = 'https://doi.org/10.1016/j.enganabound.2024.01.008'


def complex_value(value):
    z = complex(*value) if isinstance(value, list) else complex(value)
    if not np.isfinite(z):
        raise ValueError('Complex values must be finite')
    return z


def pack(values):
    values = np.asarray(values, complex)
    return np.stack((values.real, values.imag), axis=-1).tolist()


def unpack(values):
    values = np.asarray(values)
    return values[..., 0] + 1j * values[..., 1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sphere(level=2, radius=.25):
    from thermoviscous.mesh import sphere as curved_sphere
    vertices, faces = curved_sphere(level, radius)
    used, indices = np.unique(faces[:, :3], return_inverse=True)
    return vertices[used].tolist(), indices.reshape(-1, 3).tolist()


def box(size=(1, .4, .3), divisions=(8, 4, 3)):
    vertices, elements, lookup, sides = [], [], {}, []
    # Cyclic axes give positive normals for the upper faces.
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for upper in (False, True):
            start = len(elements)
            for i in range(divisions[u]):
                for j in range(divisions[v]):
                    face = []
                    for a, b in ((i, j), (i+1, j), (i+1, j+1), (i, j+1)):
                        key = [0, 0, 0]
                        key[axis], key[u], key[v] = divisions[axis] if upper else 0, a, b
                        key = tuple(key)
                        if key not in lookup:
                            lookup[key] = len(vertices)
                            vertices.append([size[k] * (key[k] / divisions[k] - .5) for k in range(3)])
                        face.append(lookup[key])
                    elements.append(face if upper else face[::-1])
            sides.append(list(range(start, len(elements))))
    return vertices, elements, sides


def geometry(scene):
    vertices = np.asarray(scene['vertices'], float)
    elements = scene['elements']
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or not elements:
        raise ValueError('Expected finite vertices and a nonempty surface mesh')
    edges, used = {}, set()
    for face in elements:
        if len(face) not in (3, 4) or any(type(i) is not int or i < 0 or i >= len(vertices) for i in face) or len(set(face)) != len(face):
            raise ValueError('Elements need three or four distinct valid vertex indices')
        points = vertices[face]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        if np.linalg.norm(normal) <= 1e-12 * scale**2:
            raise ValueError('Degenerate surface element')
        if len(face) == 4 and abs(np.dot(normal, points[3] - points[0])) > 1e-9 * np.linalg.norm(normal) * scale:
            raise ValueError('NumCalc requires planar quadrilaterals')
        for a, b in zip(face, face[1:] + face[:1]):
            key = tuple(sorted((a, b)))
            edges.setdefault(key, []).append((a, b))
        used.update(face)
    if len(used) != len(vertices):
        raise ValueError('Remove unused surface vertices')
    planes = scene.get('symmetry', {}).get('planes', [0, 0, 0])
    origin = scene.get('symmetry', {}).get('origin', [0, 0, 0])
    if len(planes) != 3 or any(type(p) is not int or not 0 <= p <= 4 for p in planes) or len(origin) != 3 or not np.isfinite(origin).all():
        raise ValueError('Invalid symmetry planes')
    for key, oriented in edges.items():
        if len(oriented) == 2 and oriented[0] == oriented[1][::-1]:
            continue
        on_plane = len(oriented) == 1 and any(planes[axis] and np.allclose(vertices[list(key), axis], origin[axis], rtol=0, atol=1e-12) for axis in range(3))
        if not on_plane:
            raise ValueError('Surface must be closed and consistently wound, except edges on specified symmetry planes')
    return vertices, elements


def prepare(scene, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vertices, elements = geometry(scene)
    frequencies = np.asarray(scene.get('frequencies', [scene.get('frequency', 500)]), float)
    if frequencies.ndim != 1 or len(frequencies) == 0 or not np.isfinite(frequencies).all() or np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0):
        raise ValueError('Frequencies must be positive, finite and strictly increasing')
    c, rho = float(scene.get('sound_speed', 343)), float(scene.get('density', 1.21))
    if not np.isfinite([c, rho]).all() or min(c, rho) <= 0:
        raise ValueError('Sound speed and density must be positive')
    iterations = scene.get('max_iterations', 500)
    if type(iterations) is not int or iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    if not np.isfinite(scene.get('cluster_edge', 0)) or scene.get('cluster_edge', 0) < 0:
        raise ValueError('cluster_edge must be finite and nonnegative')
    receivers = np.asarray(scene.get('receivers', []), float).reshape(-1, 3)
    if not np.isfinite(receivers).all():
        raise ValueError('Receivers must be finite')
    n, m, nr = len(vertices), len(elements), len(receivers)
    # Match the author's non-triangulated evaluation grid convention.
    evaluation = [[n + (3*i+j) % nr for j in range(3)] for i in range((nr+2)//3)]
    nodes = np.concatenate((vertices, receivers))
    (directory / 'Nodes.txt').write_text(f'{len(nodes)}\n' + ''.join(f'{i} {x:.17g} {y:.17g} {z:.17g}\n' for i, (x,y,z) in enumerate(nodes)))
    records = [f"{i} {' '.join(map(str, face))} 0 0 0\n" for i, face in enumerate(elements)]
    records += [f"{m+i} {' '.join(map(str, face))} 2 0 1\n" for i, face in enumerate(evaluation)]
    (directory / 'Elements.txt').write_text(f'{len(records)}\n' + ''.join(records))
    curves = []

    def amplitude(value, allow_curve=True):
        if not isinstance(value, dict):
            z = complex_value(value)
            return f'{z.real:.17g} -1 {z.imag:.17g} -1'
        f = np.asarray(value['frequencies'], float)
        z = np.array([complex_value(v) for v in value['values']])
        if len(f) != len(z) or len(f) < 2 or not np.isfinite(f).all() or np.any(np.diff(f) <= 0) or f[0] > frequencies[0] or f[-1] < frequencies[-1]:
            raise ValueError('Curves must have increasing finite frequencies covering the requested band')
        if not allow_curve:
            if len(frequencies) != 1:
                raise ValueError('Pressure/velocity curves require individual frequency solves')
            return amplitude([np.interp(frequencies[0], f, z.real), np.interp(frequencies[0], f, z.imag)])
        first = len(curves) + 1
        curves.extend([(f, z.real), (f, z.imag)])
        return f'1 {first} 1 {first+1}'

    boundaries = [scene.get('boundary', {'type': 'velocity', 'value': 0})] * m
    for patch in scene.get('patches', []):
        for i in patch['elements']:
            if type(i) is not int or not 0 <= i < m:
                raise ValueError('Invalid boundary patch element')
            boundaries[i] = patch
    bc, keywords = [], {'velocity': 'VELO', 'pressure': 'PRES', 'admittance': 'ADMI', 'impedance': 'IMPE'}
    for b, group in groupby(range(m), key=boundaries.__getitem__):
        indices = list(group)
        span = f'ELEM {indices[0]} TO {indices[-1]}'
        kind = b['type']
        if kind not in keywords:
            raise ValueError(f'Unsupported boundary type: {kind}')
        value = b.get('value', 0)
        if kind == 'impedance' and isinstance(value, dict):
            raise ValueError('Use an admittance curve for frequency-dependent impedance')
        bc.append(f'{span} {keywords[kind]} {amplitude(value, kind == "admittance")}')
        if kind == 'admittance' and 'velocity' in b:
            bc.append(f'{span} VELO {amplitude(b["velocity"], False)}')
    sources = {'plane': [], 'point': []}
    for source in scene.get('sources', []):
        kind = source['type']
        if kind not in sources:
            raise ValueError('Source type must be plane or point')
        xyz = np.asarray(source['direction' if kind == 'plane' else 'position'], float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('Source needs a finite 3-vector')
        if kind == 'plane':
            length = np.linalg.norm(xyz)
            if length == 0:
                raise ValueError('Plane wave direction cannot be zero')
            xyz = xyz / length
        sources[kind].append(f"{len(sources[kind])+1} {' '.join(f'{v:.17g}' for v in xyz)} {amplitude(source.get('amplitude', 1))}")
    method = {'dense': 0, 'slfmm': 1, 'mlfmm': 4}[scene.get('method', 'mlfmm')]
    direct = scene.get('solver', 'cgs') == 'direct'
    if scene.get('solver', 'cgs') not in ('cgs', 'direct') or (direct and method):
        raise ValueError('Direct solver requires method dense; otherwise select cgs')
    expansion = int(scene.get('expansion_terms', 8))
    if not 5 <= expansion <= 10:
        raise ValueError('Minimum expansion terms must be between 5 and 10')
    preconditioner = {'ilu': 0, 'scaling': 1, 'none': 2}[scene.get('preconditioner', 'scaling')]
    symmetry = scene.get('symmetry', {})
    planes = symmetry.get('planes', [0, 0, 0])
    origin = symmetry.get('origin', [0, 0, 0])
    lines = ['Mesh2HRTF 1.3.0', 'AcousticSolver NumCalc', '0 0 0 0 7 0', f'1 {len(frequencies)} 1 0 1 0 0',
             f'0 {len(frequencies)+1}', '0 0 0']
    lines += [f'{i+1} {f:.17g} 0' for i, f in enumerate(frequencies)]
    lines += [f'{1+bool(nr)} {len(records)} {len(nodes)} {expansion} {sum(p != 0 for p in planes)} 2 1 {method} {4 if direct else 0}',
              f'{len(sources["plane"])} {len(sources["point"])} {int(not scene.get("exterior", True))} {float(scene.get("cluster_edge", 0)):.17g} {preconditioner} 0 0',
              '0 0 0 0', f'{c:.17g} {rho:.17g} 1 0 0 0 0', 'NODES', 'Nodes.txt', 'ELEMENTS', 'Elements.txt']
    if symmetry:
        lines += ['SYMMETRY', ' '.join(map(str, planes)), ' '.join(map(str, origin))]
    lines += ['BOUNDARY', *bc, 'RETU']
    for kind, keyword in [('plane', 'PLANE WAVES'), ('point', 'POINT SOURCES')]:
        if sources[kind]:
            lines += [keyword, *sources[kind]]
    if curves:
        lines += ['CURVES', f'{len(curves)} {max(len(f) for f, _ in curves)}']
        for i, (f, z) in enumerate(curves):
            lines += [f'{i+1} {len(f)}'] + [f'{ff:.17g} {zz:.17g}' for ff, zz in zip(f, z)]
    lines += ['POST PROCESS', 'END']
    (directory / 'NC.inp').write_text('\n'.join(lines) + '\n')
    (directory / 'scene.json').write_text(json.dumps(scene, indent=2) + '\n')
    return frequencies, n, m


def field(path, ids, components=1):
    rows = {}
    for line in path.read_text().splitlines()[3:]:
        columns = line.split()
        if len(columns) == 1 + 2 * components:
            rows[int(columns[0])] = [complex(float(columns[1+2*k]), float(columns[2+2*k])) for k in range(components)]
    result = np.array([rows[i] for i in ids])
    if not np.isfinite(result).all():
        raise RuntimeError(f'Nonfinite field in {path}')
    return result[:, 0] if components == 1 else result


def run(scene, directory, backend='cpu', reference=False, benchmark=0):
    directory = Path(directory).resolve()
    frequencies, n, m = prepare(scene, directory)
    executable = ROOT / 'build' / ('NumCalcReference' if reference else 'NumCalcSolve')
    executable_hash = digest(executable)
    args = [str(executable)] if reference else [str(executable), str(directory), '--backend', backend]
    if benchmark and not reference:
        args += ['--benchmark', str(benchmark)]
    args += ['-nitermax', str(scene.get('max_iterations', 500))]
    start = time.perf_counter()
    with (directory / 'run.log').open('w') as log:
        subprocess.run(args, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
    outputs = []
    receiver_ids = range(n, n+len(scene.get('receivers', [])))
    for i, f in enumerate(frequencies):
        base = directory / 'be.out' / f'be.{i+1}'
        outputs.append({'frequency': float(f), 'boundary_pressure': pack(field(base/'pBoundary', range(m))),
                        'boundary_velocity': pack(field(base/'vBoundary', range(m))),
                        'receiver_pressure': pack(field(base/'pEvalGrid', receiver_ids)) if receiver_ids else [],
                        'receiver_velocity': pack(field(base/'vEvalGrid', receiver_ids, 3)) if receiver_ids else []})
    report = {'paper': PAPER, 'upstream_commit': subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
              'executable_sha256': executable_hash, 'backend': 'upstream' if reference else backend,
              'input_sha256': {name: digest(directory/name) for name in ('NC.inp', 'Nodes.txt', 'Elements.txt')},
              'wall_seconds': time.perf_counter()-start, 'time_convention': 'exp(-i omega t)', 'fields': outputs}
    if not reference:
        report['native'] = json.loads((directory/'native.json').read_text())
    (directory/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
