import decorators
from mesh_data import MeshData
from fast_func import fast_for, filter_global
from mesh_io import set_trimesh
from mesh_normalize import remesh
from filter_io import remove_nan_inf_model


@decorators.time_func
@decorators.cache_result
def get_all_meshes(meshes: list[MeshData]) -> list[MeshData]:
    # load the mesh of every .obj file
    new_meshes = fast_for(meshes, set_trimesh)
    return new_meshes


@decorators.time_func
@decorators.cache_result
def remesh_all_meshes(meshes: list[MeshData], target_min: int, target_max: int) -> list[MeshData]:
    new_meshes = fast_for(meshes, remesh, target_min, target_max)
    return new_meshes


@decorators.time_func
@decorators.cache_result
def remove_nan_inf_models(meshes: list[MeshData]) -> list[MeshData]:
    new_meshes = filter_global(
        meshes,
        remove_nan_inf_model,
    )
    return new_meshes
