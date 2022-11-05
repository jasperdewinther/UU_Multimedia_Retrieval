from multiprocessing.dummy import Array
from turtle import distance
from xml.sax.handler import feature_external_ges
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from numpy import float32
from mesh_data import (
    MeshData,
    get_database_as_feature_matrix,
    get_database_as_standardized_feature_matrix,
    get_feature_vector,
    get_standard_feature_vec,
)
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
    distances = get_distances(mesh_1, mesh_2)
    return np.sum(distances)


def get_distances(mesh_1: ArrayLike, mesh_2: ArrayLike) -> ArrayLike:
    simple = 5
    hist = (mesh_1.shape[0] - simple) / 5
    index_0 = int(simple + hist * 0)
    index_1 = int(simple + hist * 1)
    index_2 = int(simple + hist * 2)
    index_3 = int(simple + hist * 3)
    index_4 = int(simple + hist * 4)
    index_5 = int(simple + hist * 5)

    distances = np.abs(
        np.hstack(
            [
                (mesh_1[:simple] - mesh_2[:simple]) * 5,
                emd_np(mesh_1[index_0:index_1], mesh_2[index_0:index_1]),
                emd_np(mesh_1[index_1:index_2], mesh_2[index_1:index_2]),
                emd_np(mesh_1[index_2:index_3], mesh_2[index_2:index_3]),
                emd_np(mesh_1[index_3:index_4], mesh_2[index_3:index_4]),
                emd_np(mesh_1[index_4:index_5], mesh_2[index_4:index_5]),
            ]
        )
    )
    return distances


@decorators.time_func
@decorators.cache_result
def create_knn_structure(meshes: list[MeshData], k: int) -> NearestNeighbors:
    feature_matrix = get_database_as_standardized_feature_matrix(meshes)

    # neigh = NearestNeighbors(n_neighbors=k, metric=mesh_naive_distance)
    neigh = NearestNeighbors(n_neighbors=k, metric=mesh_distance)
    neigh.fit(feature_matrix)

    return neigh


def query_knn(
    mesh: MeshData, meshes: list[MeshData], knn: NearestNeighbors, mean: ArrayLike, std: ArrayLike
) -> list[tuple[MeshData, float, ArrayLike]]:
    feature_vector = get_standard_feature_vec(mesh, mean, std)
    feature_vector = feature_vector.reshape(1, -1)
    distances, indices = knn.kneighbors(feature_vector)
    distances = distances.reshape(-1)
    indices = indices.reshape(-1)
    results = []
    for i in range(len(indices)):
        vec_other = get_standard_feature_vec(meshes[indices[i]], mean, std)
        results.append(
            (meshes[indices[i]], distances[i], get_distances(get_standard_feature_vec(mesh, mean, std), vec_other))
        )
    return results


def show_distances(query_data: list[tuple[MeshData, float, ArrayLike]]):
    dict_form = {data[0].filename: data[2] for data in query_data}
    fig, ax = plt.subplots()
    bar_plot(ax, dict_form, total_width=0.8, single_width=0.9)
    plt.show()
    plt.clf()


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
    data_width = len(next(iter(data.items()))[1])
    plt.xticks(
        np.arange(data_width) + bar_width,
        ["surface_area", "compactness", "rectangularity", "diameter", "eccentricity", "A3", "D1", "D2", "D3", "D4"],
    )
    # Draw legend if we need
    ax.legend(bars, data.keys())
