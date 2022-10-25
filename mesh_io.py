import os
import trimesh
import decorators
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
