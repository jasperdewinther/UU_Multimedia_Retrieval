from numpy.typing import ArrayLike
from numpy import float32
from mesh_data import MeshData, get_database_as_feature_matrix, get_feature_vector
import math
import decorators

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def mesh_distance(mesh_1: ArrayLike, mesh_2: ArrayLike) -> float32:
    return math.dist(mesh_1, mesh_2)


@decorators.time_func
@decorators.cache_result
def create_knn_structure(meshes: list[MeshData], k: int) -> NearestNeighbors:
    feature_matrix = get_database_as_feature_matrix(meshes)

    neigh = NearestNeighbors(n_neighbors=k, metric=mesh_distance)
    neigh.fit(feature_matrix)

    return neigh


def query_knn(mesh: MeshData, meshes: list[MeshData], knn: NearestNeighbors, k: int) -> list[MeshData, float]:
    feature_vector = get_feature_vector(mesh)
    distances, indices = knn.kneighbors(feature_vector, k)
    results = []
    for i in range(len(indices)):
        results.append(meshes[indices[i]], distances[i])
    return results
