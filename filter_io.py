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
    volume_tri = mesh.trimesh_data.volume

    #our volume calculations
    triangles = mesh.trimesh_data.vertices.view(np.ndarray)[mesh.trimesh_data.faces]
    triangles = triangles - mesh.trimesh_data.centroid
    crossed = np.cross(triangles[:, 0, :], triangles[:, 1, :])
    volume = (
        np.sum(
            triangles[:, 2, 0] * crossed[:, 0] + triangles[:, 2, 1] * crossed[:, 1] + triangles[:, 2, 2] * crossed[:, 2]
        )
        / 6
    )

    area = mesh.trimesh_data.area
    return (
        volume_tri > 0
        and volume_tri != math.nan
        and abs(volume_tri) != math.inf
        and volume > 0
        and volume != math.nan
        and abs(volume) != math.inf
        and area > 0
        and area != math.nan
        and abs(area) != math.inf
    )


def get_broken_faces_fraction(mesh: MeshData) -> float:
    return mesh.broken_faces_count / mesh.face_count


def get_face_count(mesh: MeshData) -> float:
    return mesh.face_count
