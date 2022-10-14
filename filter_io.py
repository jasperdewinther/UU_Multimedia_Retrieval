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
def remove_degenerate_models(meshes: list[MeshData], broken_face_fraction: float) -> list[MeshData]:
    i = 0
    broken_faces = [mesh.broken_faces_count /
                    mesh.face_count for mesh in meshes]
    broken_faces.sort()
    while i < len(meshes):
        # remove meshes which contain nan or inf values
        if meshes[i].trimesh_data.volume <= 0 or meshes[i].trimesh_data.volume == math.nan or abs(meshes[i].trimesh_data.volume) == math.inf:
            meshes.pop(i)
            continue
        if meshes[i].trimesh_data.area <= 0 or meshes[i].trimesh_data.area == math.nan or abs(meshes[i].trimesh_data.area) == math.inf:
            meshes.pop(i)
            continue
        if meshes[i].broken_faces_count / meshes[i].face_count > broken_faces[int(len(broken_faces)*broken_face_fraction)]:
            meshes.pop(i)
            continue
        i += 1
    return meshes
