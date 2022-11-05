from enum import Enum
import math
from matplotlib import pyplot as plt
from pyrender import Scene, Viewer, Mesh, PerspectiveCamera, Node
import numpy as np
from sklearn.manifold import TSNE
import trimesh
from mesh_data import MeshData, get_database_as_standardized_feature_matrix
import colorsys
import PySimpleGUI as sg

from query import mesh_distance


class RenderMode(Enum):
    INORDER = 0
    TSNE = 1


def render_meshes(meshes: list[MeshData], method: RenderMode) -> Viewer:
    scene = Scene()
    if method == RenderMode.INORDER:
        build_scene(meshes, scene)
    elif method == RenderMode.TSNE:
        build_scene_tsne(meshes, scene)
    # run renderer async
    viewer = Viewer(scene, run_in_thread=False, use_raymond_lighting=True, viewport_size=(512, 512))
    return viewer


def build_scene(meshes: list[MeshData], scene: Scene):
    columns_and_rows = math.ceil(math.sqrt(len(meshes)))

    for i in range(len(meshes)):
        # add all the meshes and position them in a grid
        mesh = Mesh.from_trimesh(meshes[i].trimesh_data)
        node = scene.add(mesh)
        position = np.eye((4))
        position[:3, 3] = [
            (i % columns_and_rows) - columns_and_rows / 2,
            columns_and_rows / 2 - (i / columns_and_rows),
            -5,
        ]
        node.matrix = position

    # set the camera settings
    cam = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1)
    nc = Node(camera=cam, matrix=np.eye(4))
    scene.add_node(nc)


def build_scene_tsne(meshes: list[MeshData], scene: Scene):
    std_data = get_database_as_standardized_feature_matrix(meshes)

    X_embedded = TSNE(n_components=2, learning_rate="auto", init="random", metric=mesh_distance).fit_transform(std_data)

    classes = [*(mesh.mesh_class for mesh in meshes)]

    for i in range(len(meshes)):
        # add all the meshes and position them in a grid
        class_index = classes.index(meshes[i].mesh_class)
        color = np.array(colorsys.hsv_to_rgb((class_index / len(classes)), 1, 1))
        vertex_colors = np.zeros((meshes[i].vertex_count, 3))
        for j in range(len(vertex_colors)):
            vertex_colors[j] = color
        colored_mesh = trimesh.Trimesh(
            meshes[i].trimesh_data.vertices, meshes[i].trimesh_data.faces, vertex_colors=vertex_colors
        )
        mesh = Mesh.from_trimesh(colored_mesh)
        node = scene.add(mesh)
        position = np.eye((4))
        position[:3, 3] = [
            X_embedded[i][0],
            X_embedded[i][1],
            -5,
        ]
        node.matrix = position

    # set the camera settings
    cam = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1)
    nc = Node(camera=cam, matrix=np.eye(4))
    scene.add_node(nc)
