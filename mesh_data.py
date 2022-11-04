import math
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trimesh import Trimesh
import os
import decorators


class MeshData:
    filename: str
    mesh_class: str
    trimesh_data: Trimesh
    bounding_box: list[float]  # [x_min, y_min, z_min, x_max, y_max, z_max]
    vertex_count: int
    face_count: int
    surface_area: float
    volume: float
    compactness: float
    aabb_volume: float
    obb_volume: float
    aaobb_volume_ratio: float
    rectangularity: float
    diameter: float
    eccentricity: float
    broken_faces_count: int
    barycenter_dist_to_origin: float
    eigenvector_offset: float #cos of angle between x-axis and major eigenvector
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
        self.filename = ""
        self.mesh_class = ""
        self.trimesh_data = None
        self.bounding_box = [0, 0, 0, 0, 0, 0]
        self.vertex_count = 0
        self.face_count = 0
        self.surface_area = 0
        self.volume = None
        self.compactness = 0
        self.aabb_volume = 0
        self.obb_volume = 0
        self.rectangularity = 0
        self.diameter = 0
        self.eccentricity = 0
        self.broken_faces_count = 0
        self.barycenter_dist_to_origin = 0
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

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"\n----------------------------------------\n\
filename: {self.filename}\n\
mesh_class: {self.mesh_class}\n\
trimesh_data: {self.trimesh_data}\n\
bounding_box: {self.bounding_box}\n\
vertex_count: {self.vertex_count}\n\
face_count: {self.face_count}\n\
surface_area: {self.surface_area}\n\
compactness: {self.compactness}\n\
aabb_volume: {self.aabb_volume}\n\
obb_volume: {self.obb_volume}\n\
rectangularity: {self.rectangularity}\n\
diameter: {self.diameter}\n\
eccentricity: {self.eccentricity}\n\
broken_faces_count: {self.broken_faces_count}\n\
barycenter_dist_to_origin: {self.barycenter_dist_to_origin}\n\
----------------------------------------\n"


pd.set_option("display.float_format", lambda x: "{:.3e}".format(x) if x > 999999 or x < 0.01 else "{:.3f}".format(x))


def summarize_data(meshes: list[MeshData], figure_filename: str = None, csv_filename: str = None):
    df = pd.DataFrame()
    for mesh in meshes:
        data = {
            "filename": [mesh.filename],
            "mesh_class": [mesh.mesh_class],
            # 'bbxmin': [mesh.bounding_box[0]],
            # 'bbymin': [mesh.bounding_box[1]],
            # 'bbzmin': [mesh.bounding_box[2]],
            # 'bbxmax': [mesh.bounding_box[3]],
            # 'bbymax': [mesh.bounding_box[4]],
            # 'bbzmax': [mesh.bounding_box[5]],
            "vertex_count": [mesh.vertex_count],
            "face_count": [mesh.face_count],
            "surface_area": [mesh.surface_area],
            "compactness": [mesh.compactness],
            "aabb_volume": [mesh.aabb_volume],
            "obb_volume": [mesh.obb_volume],
            "diameter": [mesh.diameter],
            "broken_faces_count": [mesh.broken_faces_count],
            "barycenter_dist_to_origin": [mesh.barycenter_dist_to_origin],
            "cosine_of_angle_major_eigenvector_and_x-axis": [mesh.eigenvector_offset]
        }
        df = pd.concat((pd.DataFrame.from_dict(data), df), ignore_index=True)
    print(df.describe())
    df.hist(bins=100, figsize=(20, 14))  # s is an instance of Series
    df.round(3)
    print(df.describe().to_latex())
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
        plot_points = []
        for i in range(len(mesh.A3_binsize)-1):
            plot_points.append((mesh.A3_binsize[i] + mesh.A3_binsize[i+1]) / 2) 
        plt.plot(plot_points, mesh.A3)
        #plt.stairs(mesh.A3, mesh.A3_binsize, fill=False)
    plt.savefig(f"{folder_name}/hist_{current_class}_A3.png")
    plt.clf()   


    current_class = meshes[0].mesh_class
    for mesh in meshes:
        if mesh.mesh_class != current_class:
            plt.savefig(f"{folder_name}/hist_{current_class}_D1.png")
            plt.clf()
            current_class = mesh.mesh_class
        plot_points = []
        for i in range(len(mesh.D1_binsize)-1):
            plot_points.append((mesh.D1_binsize[i] + mesh.D1_binsize[i+1]) / 2)



        plt.plot(plot_points, mesh.D1)
        #plt.stairs(mesh.D1, mesh.D1_binsize, fill=False)
    plt.savefig(f"{folder_name}/hist_{current_class}_D1.png")
    plt.clf()



    current_class = meshes[0].mesh_class
    for mesh in meshes:
        if mesh.mesh_class != current_class:
            plt.savefig(f"{folder_name}/hist_{current_class}_D2.png")
            plt.clf()
            current_class = mesh.mesh_class
        plot_points = []
        for i in range(len(mesh.D2_binsize)-1):
            plot_points.append((mesh.D2_binsize[i] + mesh.D2_binsize[i+1]) / 2)



        plt.plot(plot_points, mesh.D2)
        #plt.stairs(mesh.D2, mesh.D2_binsize, fill=False)
    plt.savefig(f"{folder_name}/hist_{current_class}_D2.png")
    plt.clf()



    current_class = meshes[0].mesh_class
    for mesh in meshes:
        if mesh.mesh_class != current_class:
            plt.savefig(f"{folder_name}/hist_{current_class}_D3.png")
            plt.clf()
            current_class = mesh.mesh_class
        plot_points = []
        for i in range(len(mesh.D3_binsize)-1):
            plot_points.append((mesh.D3_binsize[i] + mesh.D3_binsize[i+1]) / 2)



        plt.plot(plot_points, mesh.D3)
        #plt.stairs(mesh.D3, mesh.D3_binsize, fill=False)
    plt.savefig(f"{folder_name}/hist_{current_class}_D3.png")
    plt.clf()



    current_class = meshes[0].mesh_class
    for mesh in meshes:
        if mesh.mesh_class != current_class:
            plt.savefig(f"{folder_name}/hist_{current_class}_D4.png")
            plt.clf()
            current_class = mesh.mesh_class
        plot_points = []
        for i in range(len(mesh.D4_binsize)-1):
            plot_points.append((mesh.D4_binsize[i] + mesh.D4_binsize[i+1]) / 2)



        plt.plot(plot_points, mesh.D4)
        #plt.stairs(mesh.D4, mesh.D4_binsize, fill=False)
    plt.savefig(f"{folder_name}/hist_{current_class}_D4.png")
    plt.clf()


def get_feature_vector(mesh: MeshData) -> ArrayLike:
    vec = np.hstack(
        [
            mesh.surface_area,
            mesh.compactness,
            mesh.rectangularity,
            mesh.diameter,
            mesh.eccentricity,
            mesh.A3,
            mesh.D1,
            mesh.D2,
            mesh.D3,
            mesh.D4,
        ]
    )
    vec = vec.astype("float32")
    return vec


def get_database_as_feature_matrix(meshes: list[MeshData]) -> ArrayLike:
    size_x = len(meshes)
    size_y = len(get_feature_vector(meshes[0]))
    feature_matrix = np.zeros((size_x, size_y))
    for i, mesh in enumerate(meshes):
        feature_matrix[i, :] = get_feature_vector(mesh)
    return feature_matrix


def get_median_mesh(meshes: list[MeshData], member: str) -> MeshData:
    values = [getattr(mesh, member) for mesh in meshes]
    return meshes[np.argsort(values)[len(values) // 2]]


def get_outlier_high_mesh(meshes: list[MeshData], member: str) -> MeshData:
    values = [getattr(mesh, member) for mesh in meshes]
    return meshes[np.argmax(values)]


def get_outlier_low_mesh(meshes: list[MeshData], member: str) -> MeshData:
    values = [getattr(mesh, member) for mesh in meshes]
    return meshes[np.argmin(values)]
