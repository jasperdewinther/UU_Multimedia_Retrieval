from mesh_data import MeshData
from trimesh import Trimesh
import numpy as np
from numpy.typing import ArrayLike
import decorators


def GetBaryCenter(mesh: Trimesh) -> ArrayLike:
    print(mesh.center_mass)
    return mesh.center_mass


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

    return meshes


def GetBoundingBoxBiggestAxis(boundingbox: list[float]) -> float:
    Dx = abs(boundingbox[3] - boundingbox[0])
    Dy = abs(boundingbox[4] - boundingbox[1])
    Dz = abs(boundingbox[5] - boundingbox[2])

    return max(Dx, Dy, Dz)

@decorators.time_func
@decorators.cache_result
def NormalizeScales(meshes: list[MeshData]) -> list[MeshData]:
    for mesh in meshes:
        scale_factor = 1 / GetBoundingBoxBiggestAxis(mesh.bounding_box)
        for vertex in mesh.trimesh_data.vertices:
            vertex *= scale_factor

    return meshes


def GetEigenValuesAndVectors(mesh: Trimesh) -> tuple[ArrayLike, ArrayLike]:
    n_points = len(mesh.vertices)
    x_coords = [vertex[0] for vertex in mesh.vertices]
    y_coords = [vertex[1] for vertex in mesh.vertices]
    z_coords = [vertex[2] for vertex in mesh.vertices]
    A = np.zeros((3, n_points))
    A[0] = x_coords
    A[1] = y_coords
    A[2] = z_coords

    A_cov = np.cov(A)

    eigenvalues, eigenvectors = np.linalg.eig(A_cov)

    return eigenvalues, eigenvectors
