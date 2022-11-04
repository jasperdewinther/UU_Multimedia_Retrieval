import decorators
from descriptors import gen_histograms, get_global_descriptor, get_minmax_shape_properties
from mesh_data import MeshData
from fast_func import fast_for, filter_fraction, filter_global
from mesh_io import set_trimesh
from mesh_normalize import remesh
from filter_io import get_broken_faces_fraction, get_face_count, remove_nan_inf_model
from normalization import NormalizeAlignment, NormalizeScale, NormalizeTranslation, NormalizeFlip


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
    new_meshes = filter_global(meshes, remove_nan_inf_model)
    return new_meshes


@decorators.time_func
@decorators.cache_result
def NormalizeTranslations(meshes: list[MeshData]) -> list[MeshData]:
    new_meshes = fast_for(meshes, NormalizeTranslation)
    return new_meshes


@decorators.time_func
@decorators.cache_result
def NormalizeScales(meshes: list[MeshData]) -> list[MeshData]:
    new_meshes = fast_for(meshes, NormalizeScale)
    return new_meshes


@decorators.time_func
@decorators.cache_result
def NormalizeAlignments(meshes: list[MeshData]) -> list[MeshData]:
    new_meshes = fast_for(meshes, NormalizeAlignment)
    return new_meshes


@decorators.time_func
@decorators.cache_result
def NormalizeFlips(meshes: list[MeshData]) -> list[MeshData]:
    new_meshes = fast_for(meshes, NormalizeFlip)
    return new_meshes


@decorators.time_func
@decorators.cache_result
def get_global_descriptors(
    meshes: list[MeshData], descriptor_iterations: int, d1_iterations: int
) -> tuple[list[MeshData], list[float]]:
    new_meshes = fast_for(meshes, get_global_descriptor, descriptor_iterations, d1_iterations)
    minmax = get_minmax_shape_properties(new_meshes)
    new_meshes = fast_for(new_meshes, gen_histograms, minmax, descriptor_iterations)
    return new_meshes, minmax


@decorators.time_func
@decorators.cache_result
def remove_models_with_holes(meshes: list[MeshData], broken_face_fraction: float) -> list[MeshData]:
    new_meshes = filter_fraction(meshes, get_broken_faces_fraction, broken_face_fraction)
    return new_meshes


@decorators.time_func
@decorators.cache_result
def remove_models_with_too_many_faces(meshes: list[MeshData], face_count_fraction: float) -> list[MeshData]:
    new_meshes = filter_fraction(meshes, get_face_count, face_count_fraction)
    return new_meshes
