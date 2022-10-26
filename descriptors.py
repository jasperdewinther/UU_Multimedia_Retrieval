import random
from tkinter import E
import trimesh
from mesh_data import MeshData
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
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


def get_global_descriptor(mesh: MeshData, descriptor_iterations: int) -> MeshData:
    # finds the surface area, compactness, axis-aligned bounding-box volume, diameter and eccentricity
    class_shape = get_class(mesh.filename)
    faces, vertices = get_faces_vertices(mesh.trimesh_data)
    bounding_box = get_bounding_box(mesh.trimesh_data)

    # surface area
    surface_area = mesh.trimesh_data.area

    # compactness

    triangles = mesh.trimesh_data.vertices.view(np.ndarray)[mesh.trimesh_data.faces]
    triangles = triangles - mesh.trimesh_data.centroid
    crossed = np.cross(triangles[:, 0, :], triangles[:, 1, :])
    volume = np.sum(
        triangles[:, 2, 0] * crossed[:, 0] + triangles[:, 2, 1] * crossed[:, 1] + triangles[:, 2, 2] * crossed[:, 2]
    )

    compactness = (surface_area**3) / (36 * math.pi * (volume**2))

    # axis-aligned bounding-box volume
    aabb_volume = mesh.trimesh_data.bounding_box.volume
    obb_volume = mesh.trimesh_data.bounding_box_oriented.volume
    rectangularity = abs(volume / obb_volume)

    # diameter according to https://stackoverflow.com/a/60955825
    hull = ConvexHull(mesh.trimesh_data.vertices)
    hullpoints = mesh.trimesh_data.vertices[hull.vertices, :]
    hdist = cdist(hullpoints, hullpoints, metric="euclidean")
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    diameter = math.dist(hullpoints[bestpair[0]], hullpoints[bestpair[1]])

    # eccentricity
    eigenvalues = mesh.trimesh_data.principal_inertia_components
    eccentricity = max(eigenvalues) - min(eigenvalues)

    # shape properties
    mesh.mesh_class = class_shape
    A3, D1, D2, D3, D4 = get_shape_properties(mesh, descriptor_iterations)

    mesh.broken_faces_count = len(trimesh.repair.broken_faces(mesh.trimesh_data))
    mesh.trimesh_data = Trimesh(mesh.trimesh_data.vertices, mesh.trimesh_data.faces)
    mesh.vertex_count = vertices
    mesh.face_count = faces
    mesh.bounding_box = bounding_box
    mesh.surface_area = surface_area
    mesh.compactness = compactness
    mesh.aabb_volume = aabb_volume
    mesh.obb_volume = obb_volume
    mesh.rectangularity = rectangularity
    mesh.diameter = diameter
    mesh.eccentricity = eccentricity
    mesh.barycenter_dist_to_origin = math.dist([0, 0, 0], mesh.trimesh_data.centroid)
    mesh.A3 = A3
    mesh.D1 = D1
    mesh.D2 = D2
    mesh.D3 = D3
    mesh.D4 = D4

    return mesh


def get_minmax_shape_properties(meshes: list[MeshData]) -> list[float]:
    minA3 = np.inf
    maxA3 = -np.inf
    minD1 = np.inf
    maxD1 = -np.inf
    minD2 = np.inf
    maxD2 = -np.inf
    minD3 = np.inf
    maxD3 = -np.inf
    minD4 = np.inf
    maxD4 = -np.inf
    for mesh in meshes:
        minA3 = min(np.amin(mesh.A3), minA3)
        maxA3 = max(np.amax(mesh.A3), maxA3)
        minD1 = min(np.amin(mesh.D1), minD1)
        maxD1 = max(np.amax(mesh.D1), maxD1)
        minD2 = min(np.amin(mesh.D2), minD2)
        maxD2 = max(np.amax(mesh.D2), maxD2)
        minD3 = min(np.amin(mesh.D3), minD3)
        maxD3 = max(np.amax(mesh.D3), maxD3)
        minD4 = min(np.amin(mesh.D4), minD4)
        maxD4 = max(np.amax(mesh.D4), maxD4)
    return [minA3, maxA3, minD1, maxD1, minD2, maxD2, minD3, maxD3, minD4, maxD4]


def gen_histograms(mesh: MeshData, min_max: list[float], descriptor_iterations: int):
    minA3 = min_max[0]
    maxA3 = min_max[1]
    minD1 = min_max[2]
    maxD1 = min_max[3]
    minD2 = min_max[4]
    maxD2 = min_max[5]
    minD3 = min_max[6]
    maxD3 = min_max[7]
    minD4 = min_max[8]
    maxD4 = min_max[9]

    counts, bin_sizes = np.histogram(
        mesh.A3, math.floor(descriptor_iterations ** (1 / 2)), [math.floor(minA3), math.ceil(maxA3)]
    )
    mesh.A3 = counts
    mesh.A3_binsize = bin_sizes

    counts, bin_sizes = np.histogram(
        mesh.D1, math.floor(descriptor_iterations ** (1 / 2)), [math.floor(minD1), math.ceil(maxD1)]
    )
    mesh.D1 = counts
    mesh.D1_binsize = bin_sizes

    counts, bin_sizes = np.histogram(
        mesh.D2, math.floor(descriptor_iterations ** (1 / 2)), [math.floor(minD2), math.ceil(maxD2)]
    )
    mesh.D2 = counts
    mesh.D2_binsize = bin_sizes

    counts, bin_sizes = np.histogram(
        mesh.D3, math.floor(descriptor_iterations ** (1 / 2)), [math.floor(minD3), math.ceil(maxD3)]
    )
    mesh.D3 = counts
    mesh.D3_binsize = bin_sizes

    counts, bin_sizes = np.histogram(
        mesh.D4, math.floor(descriptor_iterations ** (1 / 2)), [math.floor(minD4), math.ceil(maxD4)]
    )
    mesh.D4 = counts
    mesh.D4_binsize = bin_sizes
    return mesh


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


def get_shape_properties(
    mesh: MeshData, iterations: int
) -> Union[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    # finds A3, D1, D2, D3, D4

    N = len(mesh.trimesh_data.vertices) - 1

    n = iterations
    A3 = np.zeros(n)
    D1 = np.zeros(n)
    D2 = np.zeros(n)
    D3 = np.zeros(n)
    D4 = np.zeros(n)

    # A3: angle between 3 random vertices
    # compute number of samples along each of the three dimensions
    for i in range(n):
        vi = random.randint(0, N)
        vj = random.randint(0, N)
        vl = random.randint(0, N)
        if vj == vi or vj == vl or vi == vl:
            i -= 1
            continue  # do not allow duplicate points

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
        A3[i] = angle

    # D1: distance between barycenter and random vertex
    # n = N
    for i in range(n):
        vi = random.randint(0, N)

        vertexi = mesh.trimesh_data.vertices[vi]

        distance_bc = abs(math.dist(vertexi, mesh.trimesh_data.centroid))  # barycenter?
        D1[i] = distance_bc

    # D2: distance between 2 random vertices
    # compute number of samples along each of the two dimensions
    for i in range(n):
        vi = random.randint(0, N)
        vj = random.randint(0, N)
        if vj == vi:
            i -= 1
            continue  # do not allow duplicate points

        vertexi = mesh.trimesh_data.vertices[vi]
        vertexj = mesh.trimesh_data.vertices[vj]

        distance_vertices = abs(math.dist(vertexi, vertexj))
        D2[i] = distance_vertices

    # D3: square root of area of triangle given by 3 random vertices
    # compute number of samples along each of the three dimensions
    for i in range(n):
        vi = random.randint(0, N)
        vj = random.randint(0, N)
        vl = random.randint(0, N)
        if vl == vj or vl == vi or vj == vi:
            i -= 1
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

        semiperimeter = (vector1_norm + vector2_norm + vector3_norm) / 2
        area = (
            semiperimeter
            * (semiperimeter - vector1_norm)
            * (semiperimeter - vector2_norm)
            * (semiperimeter - vector3_norm)
        ) ** (1 / 2)
        D3[i] = area ** (1 / 2)

    # D4: cube root of volume of tetrahedron formed by 4 random vertices
    # compute number of samples along each of the three dimensions
    for i in range(n):
        vi = random.randint(0, N)
        vj = random.randint(0, N)
        vl = random.randint(0, N)
        vm = random.randint(0, N)
        if vm == vl or vm == vj or vm == vi or vj == vi or vl == vj or vl == vi:
            i -= 1
            continue

        vertexi = mesh.trimesh_data.vertices[vi]
        vertexj = mesh.trimesh_data.vertices[vj]
        vertexl = mesh.trimesh_data.vertices[vl]
        vertexm = mesh.trimesh_data.vertices[vm]

        volume = np.abs(np.dot(vertexi - vertexm, np.cross(vertexj - vertexm, vertexl - vertexm))) / 6
        D4[i] = volume ** (1 / 3)

    A3 = A3[~np.isnan(A3)]
    D1 = D1[~np.isnan(D1)]
    D2 = D2[~np.isnan(D2)]
    D3 = D3[~np.isnan(D3)]
    D4 = D4[~np.isnan(D4)]

    return A3, D1, D2, D3, D4
    counts, bin_sizes = np.histogram(A3, 10)
    plt.stairs(counts, bin_sizes, fill=False)
    # plt.savefig(f"histograms/{mesh.mesh_class}_A3.png")
    # plt.clf()
    counts, bin_sizes = np.histogram(D1, 10)
    plt.stairs(counts, bin_sizes, fill=False)
    # plt.savefig(f"histograms/{mesh.mesh_class}_D1.png")
    # plt.clf()
    counts, bin_sizes = np.histogram(D2, 10)
    plt.stairs(counts, bin_sizes, fill=False)
    # plt.savefig(f"histograms/{mesh.mesh_class}_D2.png")
    # plt.clf()
    counts, bin_sizes = np.histogram(D3, 10)
    plt.stairs(counts, bin_sizes, fill=False)
    # plt.savefig(f"histograms/{mesh.mesh_class}_D3.png")
    # plt.clf()
    counts, bin_sizes = np.histogram(D4, 10)
    plt.stairs(counts, bin_sizes, fill=False)
    # plt.savefig(f"histograms/{mesh.mesh_class}_D4.png")
    # plt.clf()
    # print(A3)
    # print(D1)
    # print(D2)
    # print(D3)
    # print(D4)
    # exit()
