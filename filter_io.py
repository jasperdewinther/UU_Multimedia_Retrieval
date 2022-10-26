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


def get_broken_faces_fraction(mesh: MeshData) -> float:
    return mesh.broken_faces_count / mesh.face_count


def get_face_count(mesh: MeshData) -> float:
    return mesh.face_count
