from trimesh import Trimesh
import os
from typing import Union
import csv
import numpy as np
import decorators
from mesh_data import MeshData
import math
import trimesh


def remove_nan_inf_model(mesh: MeshData) -> bool:
    volume = mesh.trimesh_data.volume
    area = mesh.trimesh_data.area
    return (
        volume > 0
        and volume != math.nan
        and abs(volume) != math.inf
        and area > 0
        and area != math.nan
        and abs(area) != math.inf
    )


@decorators.time_func
@decorators.cache_result
def remove_models_with_holes(meshes: list[MeshData], broken_face_fraction: float) -> list[MeshData]:
    i = 0
    broken_faces = [mesh.broken_faces_count / mesh.face_count for mesh in meshes]
    broken_faces.sort()
    while i < len(meshes):
        # remove meshes which contain nan or inf values
        if (
            meshes[i].broken_faces_count / meshes[i].face_count
            > broken_faces[int((len(broken_faces) - 1) * broken_face_fraction)]
        ):
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
        if meshes[i].face_count > face_counts[int((len(face_counts) - 1) * face_count_fraction)]:
            meshes.pop(i)
            continue
        i += 1
    return meshes
