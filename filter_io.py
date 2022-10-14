from trimesh import Trimesh
import os
from typing import Union
import csv
import numpy as np
import decorators
from mesh_data import MeshData
import math
import trimesh


@decorators.time_func
@decorators.cache_result
def remove_nan_inf_models(meshes: list[MeshData]) -> list[MeshData]:
    i = 0
    while i < len(meshes):
        # remove meshes which contain nan or inf values
        if meshes[i].trimesh_data.volume <= 0 or meshes[i].trimesh_data.volume == math.nan or abs(meshes[i].trimesh_data.volume) == math.inf:
            meshes.pop(i)
            continue
        if meshes[i].trimesh_data.area <= 0 or meshes[i].trimesh_data.area == math.nan or abs(meshes[i].trimesh_data.area) == math.inf:
            meshes.pop(i)
            continue
        i += 1
    return meshes


@decorators.time_func
@decorators.cache_result
def remove_models_with_holes(meshes: list[MeshData], broken_face_fraction: float) -> list[MeshData]:
    i = 0
    broken_faces = [mesh.broken_faces_count /
                    mesh.face_count for mesh in meshes]
    broken_faces.sort()
    while i < len(meshes):
        # remove meshes which contain nan or inf values
        if meshes[i].broken_faces_count / meshes[i].face_count > broken_faces[int((len(broken_faces)-1)*broken_face_fraction)]:
            meshes.pop(i)
            continue
        i += 1
    return meshes


@decorators.time_func
@decorators.cache_result
def remove_models_with_too_many_faces(meshes: list[MeshData], face_count_fraction: float) -> list[MeshData]:
    i = 0
    face_counts = [mesh.face_count for mesh in meshes]
    face_counts.sort()
    while i < len(meshes):
        # remove meshes which contain nan or inf values
        if meshes[i].face_count > face_counts[int((len(face_counts)-1)*face_count_fraction)]:
            meshes.pop(i)
            continue
        i += 1
    return meshes
