from turtle import distance
from xml.sax.handler import feature_external_ges
from numpy.typing import ArrayLike
from numpy import float32
from mesh_data import MeshData, get_database_as_feature_matrix, get_feature_vector
import math
import numpy as np
import decorators

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def mesh_naive_distance(mesh_1: ArrayLike, mesh_2: ArrayLike) -> float32:
    return math.dist(mesh_1, mesh_2)


def emd_np(a, b):
    ac = np.cumsum(a)
    bc = np.cumsum(b)
    return np.sum(np.abs(ac - bc))


def mesh_distance(mesh_1: ArrayLike, mesh_2: ArrayLike) -> float32:
    simple = 5
    hist = (mesh_1.shape[0] - simple) / 5
    index_0 = int(simple + hist * 0)
    index_1 = int(simple + hist * 1)
    index_2 = int(simple + hist * 2)
    index_3 = int(simple + hist * 3)
    index_4 = int(simple + hist * 4)
    index_5 = int(simple + hist * 5)

    distance = (
        np.sum((mesh_1[:simple] - mesh_2[:simple]) ** 2)
        + emd_np(mesh_1[index_0:index_1], mesh_2[index_0:index_1]) ** 2
        + emd_np(mesh_1[index_1:index_2], mesh_2[index_1:index_2]) ** 2
        + emd_np(mesh_1[index_2:index_3], mesh_2[index_2:index_3]) ** 2
        + emd_np(mesh_1[index_3:index_4], mesh_2[index_3:index_4]) ** 2
        + emd_np(mesh_1[index_4:index_5], mesh_2[index_4:index_5]) ** 2
    )
    return distance


@decorators.time_func
@decorators.cache_result
def create_knn_structure(meshes: list[MeshData], k: int) -> NearestNeighbors:
    feature_matrix = get_database_as_feature_matrix(meshes)

    # neigh = NearestNeighbors(n_neighbors=k, metric=mesh_naive_distance)
    neigh = NearestNeighbors(n_neighbors=k, metric=mesh_distance)
    neigh.fit(feature_matrix)

    return neigh


def query_knn(mesh: MeshData, meshes: list[MeshData], knn: NearestNeighbors, k: int) -> list[tuple[MeshData, float]]:
    feature_vector = get_feature_vector(mesh)
    feature_vector = feature_vector.reshape(1, -1)
    distances, indices = knn.kneighbors(feature_vector)
    distances = distances.reshape(-1)
    indices = indices.reshape(-1)
    results = []
    for i in range(len(indices)):
        results.append((meshes[indices[i]], distances[i]))
    return results
