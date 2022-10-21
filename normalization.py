from cgitb import small
from mesh_data import MeshData
from trimesh import Trimesh
import numpy as np
from numpy.typing import ArrayLike
import decorators
import descriptors
import math
import random


def GetBaryCenter(mesh: Trimesh) -> ArrayLike:
    return mesh.centroid


def NormalizeTranslation(mesh: Trimesh):
    baryCenter = GetBaryCenter(mesh)
    for vertex in mesh.vertices:
        vertex -= baryCenter


@decorators.time_func
@decorators.cache_result
def NormalizeTranslations(meshes: list[MeshData]) -> list[MeshData]:
    for mesh in meshes:
        baryCenter = GetBaryCenter(mesh.trimesh_data)
        for vertex in mesh.trimesh_data.vertices:
            vertex -= baryCenter

        d = math.sqrt(
            mesh.trimesh_data.centroid[0] ** 2 + mesh.trimesh_data.centroid[1] ** 2 + mesh.trimesh_data.centroid[2] ** 2)
        if (d > .01):
            print("distance", d)

    return meshes


def GetBoundingBoxBiggestAxis(boundingbox: list[float]) -> float:
    Dx = abs(boundingbox[3] - boundingbox[0])
    Dy = abs(boundingbox[4] - boundingbox[1])
    Dz = abs(boundingbox[5] - boundingbox[2])

    if (max(Dx, Dy, Dz) is 0):
        print(boundingbox)

    return max(Dx, Dy, Dz)


@decorators.time_func
@decorators.cache_result
def NormalizeScales(meshes: list[MeshData]) -> list[MeshData]:
    for mesh in meshes:
        scale_factor = 1 / \
            GetBoundingBoxBiggestAxis(
                descriptors.get_bounding_box(mesh.trimesh_data))
        for vertex in mesh.trimesh_data.vertices:
            vertex *= scale_factor

    return meshes


def GetEigenValuesAndVectors(mesh: Trimesh) -> tuple[ArrayLike, ArrayLike]:

    ## Get sum of connected face surfaces for each vertex
    ## Move vertex relative to the above away from the barycenter
    ## Calculate PCA

    samples = 10000

    x_coords = []
    y_coords = []
    z_coords = []

    total_area = 0
    triangles = []
    for face in mesh.faces:
        v0 = mesh.vertices[face[0]]
        v1 = mesh.vertices[face[1]]
        v2 = mesh.vertices[face[2]]
        #centroid = (v0 + v1 + v2) / 3

        cross = np.cross(v0-v1, v0-v2)
        area = math.sqrt(cross[0] ** 2 + cross[2] ** 2 + cross[2] ** 2)  / 2
        triangles.append((v0, v1, v2, area))
        total_area += area
    
    rands = sorted([random.random() for x in range(samples)])

    area_limit = 0
    rand_index = 0
    rand_value = rands[rand_index]
    for i in range(len(triangles)):
        
        if (rand_index >= samples):
            break

        area_limit += triangles[i][3]
        while rand_value * total_area < area_limit:
            # add random point on the triangle
            r1 = random.random()
            r2 = random.random()
            new_vertex = (1 - math.sqrt(r1))*triangles[i][0] + (math.sqrt(r1)* (1 - r2))*triangles[i][1] + (r2*math.sqrt(r1))*triangles[i][2]
            x_coords.append(new_vertex[0])
            y_coords.append(new_vertex[1])
            z_coords.append(new_vertex[2])

            # go to next random sorted number
            rand_index += 1
            if (rand_index >= samples):
                break
            rand_value = rands[rand_index]

    # vi = 0
    # biggest_area = 0
    # smallest_area = np.inf
    # max_vertex_addition = 10

    # faces = []
    # x_coords = []
    # y_coords = []
    # z_coords = []

    # for face in mesh.faces:
    #     area = 0
    #     v0 = mesh.vertices[face[0]]
    #     v1 = mesh.vertices[face[1]]
    #     v2 = mesh.vertices[face[2]]
    #     centroid = (v0 + v1 + v2) / 3

    #     cross = np.cross(v0-v1, v0-v2)
    #     area = math.sqrt(cross[0] ** 2 + cross[2] ** 2 + cross[2] ** 2)  / 2

    #     faces.append((centroid, area))

    #     if (area > biggest_area):
    #         biggest_area = area
    #         #print(area)

    #     if (area < smallest_area):
    #         smallest_area = area

    # ratio = biggest_area / smallest_area
    # print(ratio)
    # for face in faces:
    #     new_vertex_count = int(np.interp(face[1], [smallest_area, biggest_area], [0, 20]))

    #     for i in range(new_vertex_count):
    #         x_coords.append(face[0][0])
    #         y_coords.append(face[0][1])
    #         z_coords.append(face[0][2])

    # print(smallest_area)
    # print(biggest_area)


    #     #print(face[0])

    # print(len(x_coords))

        #print(vertex_faces)
    # n_points = len(mesh.vertices) + len(x_coords)
    # x_coords = [vertex[0] for vertex in mesh.vertices]
    # y_coords = [vertex[1] for vertex in mesh.vertices]
    # z_coords = [vertex[2] for vertex in mesh.vertices]
    n_points = len(x_coords)
    A = np.zeros((3, n_points))
    A[0] = x_coords
    A[1] = y_coords
    A[2] = z_coords

    A_cov = np.cov(A)

    eigenvalues, eigenvectors = np.linalg.eig(A_cov)

    return eigenvalues, eigenvectors


@decorators.time_func
@decorators.cache_result
def NormalizeAlignments(meshes: list[MeshData]) -> list[MeshData]:
    for mesh in meshes:
        #print("old c: ", mesh.trimesh_data.centroid)
        eigenvalues, eigenvectors = GetEigenValuesAndVectors(mesh.trimesh_data)
        #eigenvectors = mesh.trimesh_data.principal_inertia_vectors
        #eigenvalues = mesh.trimesh_data.principal_inertia_components

        ordered_eigenvectors = []
        ordered_eigenvalues = []

        if (eigenvalues[0] > eigenvalues[1]):
            ordered_eigenvectors.append(eigenvectors[0])
            ordered_eigenvectors.append(eigenvectors[1])
            ordered_eigenvalues.append(eigenvalues[0])
            ordered_eigenvalues.append(eigenvalues[1])
        else:
            ordered_eigenvectors.append(eigenvectors[1])
            ordered_eigenvectors.append(eigenvectors[0])
            ordered_eigenvalues.append(eigenvalues[1])
            ordered_eigenvalues.append(eigenvalues[0])

        if (eigenvalues[2] > ordered_eigenvalues[0]):
            ordered_eigenvectors.insert(0, eigenvectors[2])
            ordered_eigenvalues.insert(0, eigenvalues[2])
        elif (eigenvalues[2] > ordered_eigenvalues[1]):
            ordered_eigenvectors.insert(1, eigenvectors[2])
            ordered_eigenvalues.insert(1, eigenvalues[2])
        else:
            ordered_eigenvectors.insert(2, eigenvectors[2])
            ordered_eigenvalues.insert(2, eigenvalues[2])

        for vertex in mesh.trimesh_data.vertices:
            # print("old", vertex)
            new_vertex = [0, 0, 0]
            new_vertex[0] = np.dot(vertex, ordered_eigenvectors[0])
            new_vertex[1] = np.dot(vertex, ordered_eigenvectors[1])
            new_vertex[2] = np.dot(vertex, np.cross(
                ordered_eigenvectors[0], ordered_eigenvectors[1]))

            vertex[0] = new_vertex[0]
            vertex[1] = new_vertex[1]
            vertex[2] = new_vertex[2]
            # print("new", vertex)

        #print("new c: ", mesh.trimesh_data.centroid)

    return meshes


def NormalizeAlignment(mesh: MeshData) -> MeshData:
    eigenvalues, eigenvectors = GetEigenValuesAndVectors(mesh.trimesh_data)
    #mesh.trimesh_data = mesh.trimesh_data.apply_transform(mesh.trimesh_data.principal_inertia_transform)
    #eigenvectors = mesh.trimesh_data.principal_inertia_vectors
    #eigenvalues = mesh.trimesh_data.principal_inertia_components



    
    ordered_eigenvectors = []
    ordered_eigenvalues = []

    if (eigenvalues[0] > eigenvalues[1]):
        ordered_eigenvectors.append(eigenvectors[0])
        ordered_eigenvectors.append(eigenvectors[1])
        ordered_eigenvalues.append(eigenvalues[0])
        ordered_eigenvalues.append(eigenvalues[1])
    else:
        ordered_eigenvectors.append(eigenvectors[1])
        ordered_eigenvectors.append(eigenvectors[0])
        ordered_eigenvalues.append(eigenvalues[1])
        ordered_eigenvalues.append(eigenvalues[0]) 
    if (eigenvalues[2] > ordered_eigenvalues[0]):
        ordered_eigenvectors.insert(0, eigenvectors[2])
        ordered_eigenvalues.insert(0, eigenvalues[2])
    elif (eigenvalues[2] > ordered_eigenvalues[1]):
        ordered_eigenvectors.insert(1, eigenvectors[2])
        ordered_eigenvalues.insert(1, eigenvalues[2])
    else:
        ordered_eigenvectors.insert(2, eigenvectors[2])
        ordered_eigenvalues.insert(2, eigenvalues[2])


    print("centroid: ", mesh.trimesh_data.centroid)
    print("eigenvectors: \n", eigenvectors)
    print("eigenvalues: ", eigenvalues)
    for vertex in mesh.trimesh_data.vertices:
        # print("old", vertex)
        new_vertex = [0, 0, 0]
        new_vertex[0] = np.dot(vertex, ordered_eigenvectors[0])
        new_vertex[1] = np.dot(vertex, ordered_eigenvectors[1])
        new_vertex[2] = np.dot(vertex, np.cross(
            ordered_eigenvectors[0], ordered_eigenvectors[1]))

        vertex[0] = new_vertex[0]
        vertex[1] = new_vertex[1]
        vertex[2] = new_vertex[2]
        # print("new", vertex)

    return mesh
