from mesh_data import MeshData
import pyfqmr
from trimesh import Trimesh
import numpy as np


def remesh(mesh: MeshData, target_min: int, target_max: int) -> MeshData:
    # load the mesh of every .obj file
    mesh.trimesh_data = Trimesh(mesh.trimesh_data.vertices, mesh.trimesh_data.faces)
    mesh.trimesh_data = mesh.trimesh_data.process(validate=True)
    mesh.trimesh_data.fill_holes()

    if len(mesh.trimesh_data.faces) < target_min:
        while len(mesh.trimesh_data.faces) < target_min:
            mesh.trimesh_data = mesh.trimesh_data.subdivide()
    if len(mesh.trimesh_data.faces) > target_max:
        mesh.trimesh_data = simplify_mesh(mesh.trimesh_data, target_max)

    return mesh


def simplify_mesh(mesh: Trimesh, target_max: int) -> Trimesh:
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(target_count=target_max, verbose=0)
    vertices, faces, _ = mesh_simplifier.getMesh()
    new_mesh = Trimesh(vertices, faces)
    return new_mesh


def fan_holes(mesh: Trimesh) -> Trimesh:
    # find all edges and add their count
    edges = {}
    for face in mesh.faces:
        edge1 = (min(face[0], face[1]), max(face[0], face[1]))
        edge2 = (min(face[1], face[2]), max(face[1], face[2]))
        edge3 = (min(face[2], face[0]), max(face[2], face[0]))
        edges.setdefault(edge1, 0)
        edges.setdefault(edge2, 0)
        edges.setdefault(edge3, 0)
        edges[edge1] += 1
        edges[edge2] += 1
        edges[edge3] += 1

    # remove all edges which are connected to 2 faces
    edges = {k: v for k, v in edges.items() if v == 1}

    if len(edges) == 0:
        return mesh
    # transform to be indexable by just one edge
    edges_with_connection = {}
    for edge in edges:
        if edge[0] in edges_with_connection:
            edges_with_connection[edge[0]] = edges_with_connection[edge[0]] + [edge[1]]
        else:
            edges_with_connection[edge[0]] = [edge[1]]

        if edge[1] in edges_with_connection:
            edges_with_connection[edge[1]] = edges_with_connection[edge[1]] + [edge[0]]
        else:
            edges_with_connection[edge[1]] = [edge[0]]

    circumferences = []
    # print(edges_with_connection)
    # find all paths around holes
    while len(edges_with_connection) > 0:
        start, current = next(iter(edges_with_connection.items()))
        current = current[0]
        previous = start
        edges_with_connection.pop(start)
        path = []
        path.append(start)
        while start != current or len(path) == 1:
            # try:
            path.append(current)
            candidate_next = edges_with_connection.pop(current)

            # print(current, previous, candidate_next)
            candidate_next = candidate_next[0] if candidate_next[1] == previous else candidate_next[1]
            previous = current
            current = candidate_next
            # except:
            #    print(path)
            #    exit()
        circumferences.append(path)

    to_add_vertices = []
    to_add_indices = []
    for path in circumferences:
        # find barycenter
        total_length = 0
        points_count = len(path)

        edge_centroids = []
        lengths = []
        for i in range(points_count - 1):
            vertex1 = mesh.vertices[path[i]]
            vertex2 = mesh.vertices[path[i + 1]]
            distance = np.linalg.norm(vertex1 - vertex2)
            edge_centroids.append((vertex1 + vertex2) / 2)
            lengths.append(distance)
            total_length += distance
        centroid = np.array((0, 0, 0), dtype="float64")
        for i in range(len(edge_centroids)):
            centroid += edge_centroids[i]
        centroid /= points_count

        to_add_vertices.append(centroid)

        # create fan
        for i in range(points_count - 1):
            vertex1 = path[i]
            vertex2 = path[i + 1]
            vertex3 = len(mesh.vertices) + len(to_add_vertices) - 1
            to_add_indices.append(np.array((vertex3, vertex1, vertex2)))

    new_vertices = np.concatenate((mesh.vertices, to_add_vertices), axis=0)
    new_faces = np.concatenate((mesh.faces, to_add_indices), axis=0)
    new_mesh = Trimesh(new_vertices, new_faces)
    print("fan successful")
    return new_mesh
