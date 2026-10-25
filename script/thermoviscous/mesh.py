"""Surface mesh import and the spherical example fixture."""
import numpy as np


def sphere(level=1, radius=1.0):
    # Subdivided octahedron; project both corner and edge nodes onto the sphere.
    vertices = [np.array(p, dtype=float) for p in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
    faces = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    def midpoints():
        edges = {}
        def midpoint(a,b):
            key = tuple(sorted((a,b)))
            if key not in edges:
                p = vertices[a] + vertices[b]
                edges[key] = len(vertices)
                vertices.append(p / np.linalg.norm(p))
            return edges[key]
        return midpoint
    for _ in range(level):
        mid = midpoints()
        refined = []
        for a,b,c in faces:
            ab,bc,ca = mid(a,b),mid(b,c),mid(c,a)
            refined.extend([(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)])
        faces = refined
    mid = midpoints()
    faces = [(a,b,c,mid(a,b),mid(b,c),mid(c,a)) for a,b,c in faces]
    return np.array(vertices)*radius, np.array(faces)


def mesh(path):
    if path.suffix == '.mphtxt':
        lines = path.read_text().splitlines()
        def section(marker, dtype):
            i = lines.index(marker)+1
            values = []
            while i < len(lines) and lines[i].strip():
                values.append([dtype(x) for x in lines[i].split()]); i += 1
            return np.array(values)
        vertices = section('# Mesh vertex coordinates',float)
        faces = section('# Elements',int)
        layouts={3:[0,1,2],6:[0,1,2,3,5,4],4:[0,1,3,2],8:[0,1,3,2,4,6,7,5],9:[0,1,3,2,4,7,8,5,6]}
        if faces.shape[1] not in layouts: raise ValueError('Unsupported COMSOL surface element')
        faces=faces[:,layouts[faces.shape[1]]]
    else:
        import gmsh
        gmsh.initialize()
        try:
            gmsh.open(str(path))
            ids, xyz, _ = gmsh.model.mesh.getNodes()
            vertices = np.array(xyz).reshape(-1,3)
            indices = {int(tag): i for i,tag in enumerate(ids)}
            types, _, groups = gmsh.model.mesh.getElements(2)
            faces = []
            for kind, nodes in zip(types,groups):
                counts={2:3,9:6,3:4,16:8,10:9}
                if kind not in counts: raise ValueError('Use linear/quadratic Gmsh triangles or quadrilaterals')
                faces.extend([[indices[int(i)] for i in row] for row in np.array(nodes).reshape(-1,counts[kind])])
        finally:
            gmsh.finalize()
    sizes=[len(row) for row in faces]
    used,inverse=np.unique(np.concatenate(faces),return_inverse=True)
    result=np.split(inverse,np.cumsum(sizes)[:-1])
    return vertices[used],np.array(result) if len(set(sizes))==1 else result
