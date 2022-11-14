import math
from pyrender import Scene, Viewer, Mesh, PerspectiveCamera, Node
import numpy as np
from mesh_data import MeshData

import PySimpleGUI as sg


def render_meshes(meshes: list[MeshData]) -> Viewer:
    scene = Scene()
    build_scene(meshes, scene)
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
            2* ((i % columns_and_rows) - columns_and_rows / 2),
            2* (columns_and_rows / 2 - (i / columns_and_rows)),
            -5,
        ]
        node.matrix = position
        node2 = scene.add_node(Node(name = "test", matrix = position))

    # set the camera settings
    cam = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1)
    nc = Node(camera=cam, matrix=np.eye(4))
    scene.add_node(nc)
