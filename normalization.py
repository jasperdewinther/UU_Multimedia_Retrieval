from cgitb import small
from mesh_data import MeshData
from trimesh import Trimesh
import numpy as np
from numpy.typing import ArrayLike
import decorators
import descriptors
import math
import random
import trimesh


def GetBaryCenter(mesh: Trimesh) -> ArrayLike:
    return mesh.centroid


def NormalizeTranslation(mesh: Trimesh):
    baryCenter = GetBaryCenter(mesh)
    for vertex in mesh.vertices:
        vertex -= baryCenter


def NormalizeTranslation(mesh: MeshData) -> MeshData:
    baryCenter = GetBaryCenter(mesh.trimesh_data)
    for vertex in mesh.trimesh_data.vertices:
        vertex -= baryCenter

    d = math.sqrt(
        mesh.trimesh_data.centroid[0] ** 2 + mesh.trimesh_data.centroid[1] ** 2 + mesh.trimesh_data.centroid[2] ** 2
    )
    if d > 0.01:
        print("distance", d)

    return mesh


def GetBoundingBoxBiggestAxis(boundingbox: list[float]) -> float:
    Dx = abs(boundingbox[3] - boundingbox[0])
    Dy = abs(boundingbox[4] - boundingbox[1])
    Dz = abs(boundingbox[5] - boundingbox[2])

    if max(Dx, Dy, Dz) == 0:
        print(boundingbox)

    return max(Dx, Dy, Dz)


def NormalizeScale(mesh: MeshData) -> MeshData:
    scale_factor = 1 / GetBoundingBoxBiggestAxis(descriptors.get_bounding_box(mesh.trimesh_data))
    for vertex in mesh.trimesh_data.vertices:
        vertex *= scale_factor

    return mesh


def GetEigenValuesAndVectors(mesh: Trimesh) -> tuple[ArrayLike, ArrayLike]:
    vertices = RandomlySamplePointsOverMesh(mesh, 10000)

    x_coords = [vertex[0] for vertex in vertices]
    y_coords = [vertex[1] for vertex in vertices]
    z_coords = [vertex[2] for vertex in vertices]
    n_points = len(vertices)
    A = np.zeros((3, n_points))
    A[0] = x_coords
    A[1] = y_coords
    A[2] = z_coords

    A_cov = np.cov(A)

    eigenvalues, eigenvectors = np.linalg.eig(A_cov)

    ordered_eigenvectors = []
    ordered_eigenvalues = []

    if eigenvalues[0] > eigenvalues[1]:
        ordered_eigenvectors.append(eigenvectors[0])
        ordered_eigenvectors.append(eigenvectors[1])
        ordered_eigenvalues.append(eigenvalues[0])
        ordered_eigenvalues.append(eigenvalues[1])
    else:
        ordered_eigenvectors.append(eigenvectors[1])
        ordered_eigenvectors.append(eigenvectors[0])
        ordered_eigenvalues.append(eigenvalues[1])
        ordered_eigenvalues.append(eigenvalues[0])
    if eigenvalues[2] > ordered_eigenvalues[0]:
        ordered_eigenvectors.insert(0, eigenvectors[2])
        ordered_eigenvalues.insert(0, eigenvalues[2])
    elif eigenvalues[2] > ordered_eigenvalues[1]:
        ordered_eigenvectors.insert(1, eigenvectors[2])
        ordered_eigenvalues.insert(1, eigenvalues[2])
    else:
        ordered_eigenvectors.insert(2, eigenvectors[2])
        ordered_eigenvalues.insert(2, eigenvalues[2])

    return ordered_eigenvalues, ordered_eigenvectors


def NormalizeAlignment(mesh: MeshData) -> MeshData:
    eigenvalues, eigenvectors = GetEigenValuesAndVectors(mesh.trimesh_data)
    # mesh.trimesh_data = mesh.trimesh_data.apply_transform(mesh.trimesh_data.principal_inertia_transform)
    # eigenvectors = mesh.trimesh_data.principal_inertia_vectors
    # eigenvalues = mesh.trimesh_data.principal_inertia_components

    ordered_eigenvectors = []
    ordered_eigenvalues = []

    if eigenvalues[0] > eigenvalues[1]:
        ordered_eigenvectors.append(eigenvectors[0])
        ordered_eigenvectors.append(eigenvectors[1])
        ordered_eigenvalues.append(eigenvalues[0])
        ordered_eigenvalues.append(eigenvalues[1])
    else:
        ordered_eigenvectors.append(eigenvectors[1])
        ordered_eigenvectors.append(eigenvectors[0])
        ordered_eigenvalues.append(eigenvalues[1])
        ordered_eigenvalues.append(eigenvalues[0])
    if eigenvalues[2] > ordered_eigenvalues[0]:
        ordered_eigenvectors.insert(0, eigenvectors[2])
        ordered_eigenvalues.insert(0, eigenvalues[2])
    elif eigenvalues[2] > ordered_eigenvalues[1]:
        ordered_eigenvectors.insert(1, eigenvectors[2])
        ordered_eigenvalues.insert(1, eigenvalues[2])
    else:
        ordered_eigenvectors.insert(2, eigenvectors[2])
        ordered_eigenvalues.insert(2, eigenvalues[2])

    # print("centroid: ", mesh.trimesh_data.centroid)
    # print("eigenvectors: \n", eigenvectors)
    # print("eigenvalues: ", eigenvalues)
    for vertex in mesh.trimesh_data.vertices:
        # print("old", vertex)
        new_vertex = [0, 0, 0]
        new_vertex[0] = np.dot(vertex, ordered_eigenvectors[0])
        new_vertex[1] = np.dot(vertex, ordered_eigenvectors[1])
        new_vertex[2] = np.dot(vertex, np.cross(ordered_eigenvectors[0], ordered_eigenvectors[1]))

        vertex[0] = new_vertex[0]
        vertex[1] = new_vertex[1]
        vertex[2] = new_vertex[2]
        # print("new", vertex)

    return mesh

def NormalizeFlip(mesh: MeshData) -> MeshData:
    # Fx = Fy = Fz = 0

    # for face in mesh.trimesh_data.faces:
    #     v0 = mesh.trimesh_data.vertices[face[0]]
    #     v1 = mesh.trimesh_data.vertices[face[1]]
    #     v2 = mesh.trimesh_data.vertices[face[2]]
        
    #     cross = np.cross(v0 - v1, v0 - v2)
    #     area = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) / 2

    #     centroid = (v0 + v1 + v2) / 3
    #     centroid *= area

    #     Fx += np.sign(centroid[0]) * (centroid[0] ** 2)
    #     Fy += np.sign(centroid[1]) * (centroid[1] ** 2)
    #     Fz += np.sign(centroid[2]) * (centroid[2] ** 2)

    # xScale = np.sign(Fx)
    # yScale = np.sign(Fy)
    # zScale = np.sign(Fz)
    xScale = np.sign(mesh.trimesh_data.center_mass[0])
    yScale = np.sign(mesh.trimesh_data.center_mass[1])
    zScale = np.sign(mesh.trimesh_data.center_mass[2])
    # print(mesh.trimesh_data.centroid)
    # print(mesh.trimesh_data.center_mass)
    # print(xScale, " ", yScale, " ", zScale, " ")
    # print(mesh.trimesh_data.vertices[0][1])

    for vertex in mesh.trimesh_data.vertices:
        vertex[0] *= xScale
        vertex[1] *= yScale
        vertex[2] *= zScale

    # print(mesh.trimesh_data.vertices[0][1])
    mesh.trimesh_data.fix_normals()
    return mesh

def RandomlySamplePointsOverMesh(mesh: Trimesh, sampleCount: int) -> tuple[float, float, float]:
    vertices = []

    total_area = 0
    triangles = []
    for face in mesh.faces:
        v0 = mesh.vertices[face[0]]
        v1 = mesh.vertices[face[1]]
        v2 = mesh.vertices[face[2]]
        # centroid = (v0 + v1 + v2) / 3

        cross = np.cross(v0 - v1, v0 - v2)
        area = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) / 2
        triangles.append((v0, v1, v2, area))
        total_area += area

    rands = sorted([random.random() for x in range(sampleCount)])

    area_limit = 0
    rand_index = 0
    rand_value = rands[rand_index]
    for i in range(len(triangles)):

        if rand_index >= sampleCount:
            break

        area_limit += triangles[i][3]
        while rand_value * total_area < area_limit:
            # add random point on the triangle
            r1 = random.random()
            r2 = random.random()
            new_vertex = (
                (1 - math.sqrt(r1)) * triangles[i][0]
                + (math.sqrt(r1) * (1 - r2)) * triangles[i][1]
                + (r2 * math.sqrt(r1)) * triangles[i][2]
            )
            vertices.append((new_vertex[0], new_vertex[1], new_vertex[2]))

            # go to next random sorted number
            rand_index += 1
            if rand_index >= sampleCount:
                break
            rand_value = rands[rand_index]

    return vertices
