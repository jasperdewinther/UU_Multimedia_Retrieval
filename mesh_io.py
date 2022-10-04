import os
import trimesh
import decorators
import mesh_data


@decorators.time_func
@decorators.cache_result
def get_all_obj_files(folder: str) -> list[mesh_data.MeshData]:
    # return the location of every .obj file in a directory and its subdirectories
    all_files = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".obj"):
                data = mesh_data.MeshData()
                data.filename = subdir + '/' + file
                all_files.append(data)
    return all_files


@decorators.time_func
@decorators.cache_result
def get_all_meshes(meshes: list[mesh_data.MeshData]) -> list[mesh_data.MeshData]:
    # load the mesh of every .obj file
    for file in meshes:
        file.trimesh_data = trimesh.load(file.filename)
    return meshes


def get_mesh(file_name: str) -> trimesh.Trimesh:
    return trimesh.load(file_name)
