import os
import trimesh
import decorators
from functools import wraps
import pickle


@decorators.time_func
@decorators.cache_result
def get_all_obj_files(folder):
    # return the location of every .obj file in a directory and its subdirectories
    all_files = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".obj"):
                all_files.append(subdir + '/' + file)
    return all_files


@decorators.time_func
@decorators.cache_result
def get_all_meshes(files):
    # load the mesh of every .obj file
    meshes = []
    for file in files:
        mesh = trimesh.load(file)
        meshes.append(mesh)
    return meshes
