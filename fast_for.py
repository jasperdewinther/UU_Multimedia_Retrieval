from multiprocessing import Pool, cpu_count
from typing import Callable
from mesh_data import MeshData
from typing import Union


def failing_apply(obj: MeshData, func: Callable, *arguments) -> Union[MeshData, None]:
    try:
        if len(arguments[0]) == 0:
            return func(obj)
        else:
            return func(obj, arguments)
    except:
        return None


def fast_for(meshes: list[MeshData], func: Callable, *arguments) -> list[MeshData]:
    # unfortunately is still slower than expected
    with Pool(cpu_count()) as p:
        new_list = p.starmap(failing_apply, [[mesh, func, arguments] for mesh in meshes])
    cleaned_vec = [i for i in new_list if i is not None]
    if len(meshes) != len(cleaned_vec):
        print(f"{len(meshes) - len(cleaned_vec)} meshes were removed due to errors")
    return cleaned_vec
