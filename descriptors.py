import random
import trimesh
import mesh_data
import math
import ast
from trimesh import Trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import os
import decorators
from typing import Union
import time


@decorators.time_func
@decorators.cache_result
def get_global_descriptors(meshes: list[mesh_data.MeshData]) -> list[mesh_data.MeshData]:
    # finds the surface area, compactness, axis-aligned bounding-box volume, diameter and eccentricity
    counter = 0
    for mesh in meshes:
        class_shape = get_class(mesh.filename)
        faces, vertices = get_faces_vertices(mesh.trimesh_data)
        bounding_box = get_bounding_box(mesh.trimesh_data)

        # surface area
        surface_area = mesh.trimesh_data.area

        # compactness
        volume = mesh.trimesh_data.volume
        compactness = (surface_area**3) / (36*math.pi*(volume**2))

        # axis-aligned bounding-box volume
        aabb_volume = mesh.trimesh_data.bounding_box.volume
        obb_volume = mesh.trimesh_data.bounding_box_oriented.volume

        # diameter according to https://stackoverflow.com/a/60955825
        hull = ConvexHull(mesh.trimesh_data.vertices)
        hullpoints = mesh.trimesh_data.vertices[hull.vertices, :]
        hdist = cdist(hullpoints,
                      hullpoints, metric='euclidean')
        bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
        diameter = math.dist(
            hullpoints[bestpair[0]], hullpoints[bestpair[1]])

        # eccentricity
        eigenvalues = mesh.trimesh_data.principal_inertia_components
        eccentricity = max(eigenvalues) - min(eigenvalues)

        # shape properties
        shape_properties = get_shape_properties(mesh)

        mesh.mesh_class = class_shape
        mesh.broken_faces_count = len(
            trimesh.repair.broken_faces(mesh.trimesh_data))
        mesh.trimesh_data = Trimesh(
            mesh.trimesh_data.vertices, mesh.trimesh_data.faces)
        mesh.vertex_count = vertices
        mesh.face_count = faces
        mesh.bounding_box = bounding_box
        mesh.surface_area = surface_area
        mesh.compactness = compactness
        mesh.aabb_volume = aabb_volume
        mesh.obb_volume = obb_volume
        mesh.diameter = diameter
        mesh.eccentricity = eccentricity

        counter += 1

    return meshes


def get_class(mesh_file: str) -> str:
    # finds the the class of the shape (for sheep, etc this is "assets")
    full_path = os.path.dirname(mesh_file)
    class_shape = os.path.basename(full_path)
    return class_shape


def get_faces_vertices(mesh: Trimesh) -> Union[int, int]:
    # finds number of vertices and faces of the shape and writes to csv file
    faces = mesh.faces.shape[0]
    vertices = mesh.vertices.shape[0]
    return faces, vertices


def get_bounding_box(mesh: Trimesh) -> list[float]:
    # find bounding box of the shape
    # [x_min, y_min, z_min, x_max, y_max, z_max]
    bounding_box = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]
    for vertex in mesh.vertices:
        if bounding_box[0] > vertex[0]:
            bounding_box[0] = vertex[0]
        if bounding_box[1] > vertex[1]:
            bounding_box[1] = vertex[1]
        if bounding_box[2] > vertex[2]:
            bounding_box[2] = vertex[2]
        if bounding_box[3] < vertex[0]:
            bounding_box[3] = vertex[0]
        if bounding_box[4] < vertex[1]:
            bounding_box[4] = vertex[1]
        if bounding_box[5] < vertex[2]:
            bounding_box[5] = vertex[2]
    return bounding_box


def get_shape_properties(mesh: mesh_data.MeshData) -> None:
    # finds A3, D1, D2, D3, D4

    N = len(mesh.trimesh_data.vertices) - 1

    A3 = []
    D1 = []
    D2 = []
    D3 = []
    D4 = []

    # A3: angle between 3 random vertices
    n = 10000
    # compute number of samples along each of the three dimensions
    k = int(pow(n, 1.0/3.0))
    for _ in range(k):
        # !!!!!!!!!!! lijst maken met alle vertices, als vertices gekozen verwijder hem uit de lijst
        vi = random.randint(0, N)
        for _ in range(k):
            vj = random.randint(0, N)
            if (vj == vi):
                continue  # do not allow duplicate points
            for _ in range(k):
                vl = random.randint(0, N)
                if (vl == vj or vl == vi):
                    continue

                vertexi = mesh.trimesh_data.vertices[vi]
                vertexj = mesh.trimesh_data.vertices[vj]
                vertexl = mesh.trimesh_data.vertices[vl]

                vector1 = vertexi - vertexj
                vector2 = vertexi - vertexl

                vector1_norm = np.linalg.norm(vector1)
                vector2_norm = np.linalg.norm(vector2)

                unit_vector1 = vector1 / vector1_norm
                unit_vector2 = vector2 / vector2_norm

                angle = np.arccos(np.dot(unit_vector1, unit_vector2))
                A3.append(angle)

    # D1: distance between barycenter and random vertex
    n = N
    for _ in range(k):
        vi = random.randint(0, N)

        vertexi = mesh.trimesh_data.vertices[vi]

        distance_bc = abs(
            math.dist(vertexi, mesh.trimesh_data.centroid))  # barycenter?
        D1.append(distance_bc)

    # D2: distance between 2 random vertices
    n = 10000
    # compute number of samples along each of the two dimensions
    k = int(pow(n, 1.0/2.0))
    for _ in range(k):
        vi = random.randint(0, N)
        for _ in range(k):
            vj = random.randint(0, N)
            if (vj == vi):
                continue  # do not allow duplicate points

            vertexi = mesh.trimesh_data.vertices[vi]
            vertexj = mesh.trimesh_data.vertices[vj]

            distance_vertices = abs(math.dist(vertexi, vertexj))
            D2.append(distance_vertices)

    # D3: square root of area of triangle given by 3 random vertices
    n = 10000
    # compute number of samples along each of the three dimensions
    k = int(pow(n, 1.0/3.0))
    for _ in range(k):
        vi = random.randint(0, N)
        for _ in range(k):
            vj = random.randint(0, N)
            if (vj == vi):
                continue  # do not allow duplicate points
            for _ in range(k):
                vl = random.randint(0, N)
                if (vl == vj or vl == vi):
                    continue

                vertexi = mesh.trimesh_data.vertices[vi]
                vertexj = mesh.trimesh_data.vertices[vj]
                vertexl = mesh.trimesh_data.vertices[vl]

                vector1 = vertexi - vertexj
                vector2 = vertexi - vertexl
                vector3 = vertexj - vertexl

                vector1_norm = np.linalg.norm(vector1)
                vector2_norm = np.linalg.norm(vector2)
                vector3_norm = np.linalg.norm(vector3)

                semiperimeter = (
                    vector1_norm + vector2_norm + vector3_norm) / 2
                area = (semiperimeter * (semiperimeter - vector1_norm) *
                        (semiperimeter - vector2_norm) * (semiperimeter - vector3_norm))**(1/2)
                D3.append(area**(1/2))

    # D4: cube root of volume of tetrahedron formed by 4 random vertices
    n = 10000
    # compute number of samples along each of the three dimensions
    k = int(pow(n, 1.0/4.0))
    for _ in range(k):
        vi = random.randint(0, N)
        for _ in range(k):
            vj = random.randint(0, N)
            if (vj == vi):
                continue  # do not allow duplicate points
            for _ in range(k):
                vl = random.randint(0, N)
                if (vl == vj or vl == vi):
                    continue
                for _ in range(k):
                    vm = random.randint(0, N)
                    if (vm == vl or vm == vj or vm == vi):
                        continue

                    vertexi = mesh.trimesh_data.vertices[vi]
                    vertexj = mesh.trimesh_data.vertices[vj]
                    vertexl = mesh.trimesh_data.vertices[vl]
                    vertexm = mesh.trimesh_data.vertices[vm]

                    volume = np.linalg.det(
                        np.dot(vertexi - vertexm, np.cross(vertexj - vertexm, vertexl - vertexm)))/6
                    D4.append(volume**(1/3))
    print(A3)
    print(D1)
    print(D2)
    print(D3)
    print(D4)
    exit()
