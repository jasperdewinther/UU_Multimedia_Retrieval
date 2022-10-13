from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trimesh import Trimesh


class MeshData:
    filename: str
    mesh_class: str
    trimesh_data: Trimesh
    bounding_box: list[float]  # [x_min, y_min, z_min, x_max, y_max, z_max]
    vertex_count: int
    face_count: int
    surface_area: float
    compactness: float
    bb_volume: float
    diameter: float
    broken_faces_count: int

    def __init__(self):
        self.filename = ''
        self.mesh_class = ''
        self.trimesh_data = None
        self.bounding_box = [0, 0, 0, 0, 0, 0]
        self.vertex_count = 0
        self.face_count = 0
        self.surface_area = 0
        self.compactness = 0
        self.bb_volume = 0
        self.diameter = 0
        self.broken_faces_count = 0


pd.set_option('display.float_format', lambda x: '%.5f' % x)


def summarize_data(meshes: list[MeshData], figure_filename: str = None, csv_filename: str = None):
    df = pd.DataFrame()
    for mesh in meshes:
        data = {
            'filename': [mesh.filename],
            'mesh_class': [mesh.mesh_class],
            'bbxmin': [mesh.bounding_box[0]],
            'bbymin': [mesh.bounding_box[1]],
            'bbzmin': [mesh.bounding_box[2]],
            'bbxmax': [mesh.bounding_box[3]],
            'bbymax': [mesh.bounding_box[4]],
            'bbzmax': [mesh.bounding_box[5]],
            'vertex_count': [mesh.vertex_count],
            'face_count': [mesh.face_count],
            'surface_area': [mesh.surface_area],
            'compactness': [mesh.compactness],
            'bb_volume': [mesh.bb_volume],
            'diameter': [mesh.diameter],
            'broken_faces_count': [mesh.broken_faces_count]
        }
        df = pd.concat((pd.DataFrame.from_dict(data), df), ignore_index=True)
    print(df.describe())
    df.hist(bins=100, figsize=(20, 14))  # s is an instance of Series
    if figure_filename:
        plt.savefig(figure_filename)
        plt.clf()
    if csv_filename:
        df.to_csv(csv_filename)


def generate_histogram(meshes: list[MeshData], bins: int, member: str) -> ArrayLike:
    data = np.zeros((len(meshes)))
    for i, mesh in enumerate(meshes):
        data[i] = getattr(mesh, member)
    hist = np.histogram(data, bins)
    return hist


def render_histogram(meshes: list[MeshData], bins: int, member: str, filename: str):
    data = np.zeros((len(meshes)))
    for i, mesh in enumerate(meshes):
        data[i] = getattr(mesh, member)
    counts, bin_sizes = np.histogram(data, bins)
    plt.stairs(counts, bin_sizes, fill=True)
    plt.savefig(filename)
    plt.clf()
