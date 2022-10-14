from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trimesh import Trimesh
import os


class MeshData:
    filename: str
    mesh_class: str
    trimesh_data: Trimesh
    bounding_box: list[float]  # [x_min, y_min, z_min, x_max, y_max, z_max]
    vertex_count: int
    face_count: int
    surface_area: float
    compactness: float
    aabb_volume: float
    obb_volume: float
    diameter: float
    broken_faces_count: int
    A3: ArrayLike
    A3_binsize: ArrayLike
    D1: ArrayLike
    D1_binsize: ArrayLike
    D2: ArrayLike
    D2_binsize: ArrayLike
    D3: ArrayLike
    D3_binsize: ArrayLike
    D4: ArrayLike
    D4_binsize: ArrayLike

    def __init__(self):
        self.filename = ''
        self.mesh_class = ''
        self.trimesh_data = None
        self.bounding_box = [0, 0, 0, 0, 0, 0]
        self.vertex_count = 0
        self.face_count = 0
        self.surface_area = 0
        self.compactness = 0
        self.aabb_volume = 0
        self.obb_volume = 0
        self.diameter = 0
        self.broken_faces_count = 0
        self.A3 = np.zeros(0)
        self.A3_binsize = np.zeros(0)
        self.D1 = np.zeros(0)
        self.D1_binsize = np.zeros(0)
        self.D2 = np.zeros(0)
        self.D2_binsize = np.zeros(0)
        self.D3 = np.zeros(0)
        self.D3_binsize = np.zeros(0)
        self.D4 = np.zeros(0)
        self.D4_binsize = np.zeros(0)


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
            'aabb_volume': [mesh.aabb_volume],
            'obb_volume': [mesh.obb_volume],
            'diameter': [mesh.diameter],
            'broken_faces_count': [mesh.broken_faces_count]
        }
        df = pd.concat((pd.DataFrame.from_dict(data), df), ignore_index=True)
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


def render_class_histograms(meshes: list[MeshData], folder_name: str):
    os.makedirs(os.path.dirname(folder_name), exist_ok=True)

    current_class = meshes[0].mesh_class
    for mesh in meshes:
        if mesh.mesh_class != current_class:
            plt.savefig(f"{folder_name}/hist_{current_class}_A3.png")
            plt.clf()
            current_class = mesh.mesh_class
        plt.stairs(mesh.A3, mesh.A3_binsize, fill=False)
    plt.savefig(f"{folder_name}/hist_{current_class}_A3.png")
    plt.clf()

    current_class = meshes[0].mesh_class
    for mesh in meshes:
        if mesh.mesh_class != current_class:
            plt.savefig(f"{folder_name}/hist_{current_class}_D1.png")
            plt.clf()
            current_class = mesh.mesh_class
        plt.stairs(mesh.D1, mesh.D1_binsize, fill=False)
    plt.savefig(f"{folder_name}/hist_{current_class}_D1.png")
    plt.clf()

    current_class = meshes[0].mesh_class
    for mesh in meshes:
        if mesh.mesh_class != current_class:
            plt.savefig(f"{folder_name}/hist_{current_class}_D2.png")
            plt.clf()
            current_class = mesh.mesh_class
        plt.stairs(mesh.D2, mesh.D2_binsize, fill=False)
    plt.savefig(f"{folder_name}/hist_{current_class}_D2.png")
    plt.clf()

    current_class = meshes[0].mesh_class
    for mesh in meshes:
        if mesh.mesh_class != current_class:
            plt.savefig(f"{folder_name}/hist_{current_class}_D3.png")
            plt.clf()
            current_class = mesh.mesh_class
        plt.stairs(mesh.D3, mesh.D3_binsize, fill=False)
    plt.savefig(f"{folder_name}/hist_{current_class}_D3.png")
    plt.clf()

    current_class = meshes[0].mesh_class
    for mesh in meshes:
        if mesh.mesh_class != current_class:
            plt.savefig(f"{folder_name}/hist_{current_class}_D4.png")
            plt.clf()
            current_class = mesh.mesh_class
        plt.stairs(mesh.D4, mesh.D4_binsize, fill=False)
    plt.savefig(f"{folder_name}/hist_{current_class}_D4.png")
    plt.clf()
