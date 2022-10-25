import os
from trimesh import Trimesh
import trimesh
import decorators
from fast_for import fast_for
from mesh_data import MeshData


@decorators.time_func
@decorators.cache_result
def get_all_obj_files(folder: str) -> list[MeshData]:
    # return the location of every .obj file in a directory and its subdirectories
    all_files = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".off") or file.endswith(".obj"):
                data = MeshData()
                data.filename = subdir + "/" + file
                all_files.append(data)
    return all_files


def set_trimesh(mesh: MeshData) -> MeshData:
    mesh.trimesh_data = trimesh.load(mesh.filename)
    return mesh


@decorators.time_func
@decorators.cache_result
def get_all_meshes(meshes: list[MeshData]) -> list[MeshData]:
    # load the mesh of every .obj file
    new_meshes = fast_for(meshes, set_trimesh)
    return new_meshes


def get_mesh(file_name: str) -> Trimesh:
    return trimesh.load(file_name)
