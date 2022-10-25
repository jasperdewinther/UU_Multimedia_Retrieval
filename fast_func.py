from multiprocessing import Pool, cpu_count
from typing import Callable

from numpy import float32
from mesh_data import MeshData
from typing import Union
from tqdm import tqdm


def failing_apply(obj: MeshData, func: Callable, *arguments) -> Union[MeshData, None]:
    try:
        if len(arguments) == 0:
            return func(obj)
        else:
            return func(obj, *arguments)
    except Exception as e:
        print(e)
        return None


def fast_for(meshes: list[MeshData], func: Callable, *arguments) -> list[MeshData]:
    # unfortunately is still slower than expected
    with Pool(cpu_count()) as p:
        new_list = p.starmap(failing_apply, tqdm([[mesh, func, *arguments] for mesh in meshes]))
    cleaned_vec = [i for i in new_list if i is not None]
    if len(meshes) != len(cleaned_vec):
        print(f"{len(meshes) - len(cleaned_vec)} meshes were removed due to errors")
    return cleaned_vec


# remove all meshes which return false for a certain func
def filter_global(meshes: list[MeshData], func: Callable, *arguments) -> list[MeshData]:
    return [mesh for mesh in meshes if func(mesh, *arguments)]


# throw away all values that end up in tho top fraction
def filter_fraction(meshes: list[MeshData], func: Callable, fraction: float, *arguments) -> list[MeshData]:
    values = [func(mesh, *arguments) for mesh in meshes]
    values.sort()
    cutoff = values[int((len(values) - 1) * fraction)]
    return [mesh for mesh in meshes if func(mesh, *arguments) <= cutoff]
