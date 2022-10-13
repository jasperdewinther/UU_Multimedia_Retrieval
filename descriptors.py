import trimesh
import mesh_data
import math
import ast
from trimesh import Trimesh
import numpy as np
import pandas as pd
import os
import decorators
from typing import Union


@decorators.time_func
@decorators.cache_result
def get_global_descriptors(meshes: list[mesh_data.MeshData]) -> list[mesh_data.MeshData]:
    # finds the surface area, compactness, axis-aligned boudning-box volume, diameter and eccentricity
    counter = 0
    for mesh in meshes:
        class_shape = get_class(mesh.filename)
        faces, vertices = get_faces_vertices(mesh.trimesh_data)
        bounding_box = get_bounding_box(mesh.trimesh_data)

        # surface area
        surface_area = mesh.trimesh_data.area

        # compactness
        volume = mesh.trimesh_data.volume
        compactness = (surface_area**3) / (36*math.pi*(volume**2))

        # axis-aligned bounding-box volume
        x1 = bounding_box[0]
        y1 = bounding_box[1]
        z1 = bounding_box[2]
        x2 = bounding_box[3]
        y2 = bounding_box[4]
        z2 = bounding_box[5]
        bb_volume = abs(x2 - x1) * abs(y2 - y1) * abs(z2 - z1)

        # diameter
        diameter = max(x2 - x1, y2 - y1, z2 - z1)

        # eccentricity
        eccentricity = 0

        mesh.mesh_class = class_shape
        mesh.broken_faces_count = len(
            trimesh.repair.broken_faces(mesh.trimesh_data))
        mesh.vertex_count = vertices
        mesh.face_count = faces
        mesh.bounding_box = bounding_box
        mesh.surface_area = surface_area
        mesh.compactness = compactness
        mesh.bb_volume = bb_volume
        mesh.diameter = diameter
        mesh.eccentricity = eccentricity

        counter += 1

    return meshes


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
