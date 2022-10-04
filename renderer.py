import math
import pyrender
import numpy as np
from mesh_data import MeshData

import PySimpleGUI as sg


def render_meshes(meshes: list[MeshData]):
    columns_and_rows = math.ceil(math.sqrt(len(meshes)))
    scene = pyrender.Scene()

    for i in range(len(meshes)):
        # add all the meshes and position them in a grid
        mesh = pyrender.Mesh.from_trimesh(meshes[i].trimesh_data)
        node = scene.add(mesh)
        position = np.eye((4))
        position[:3, 3] = [(i % columns_and_rows)-columns_and_rows/2,
                           (i/columns_and_rows)-columns_and_rows/2, -5]
        node.matrix = position

    # set the camera settings
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    nc = pyrender.Node(camera=cam, matrix=np.eye(4))
    scene.add_node(nc)

    # run renderer async
    viewer = pyrender.Viewer(
        scene, run_in_thread=True, use_raymond_lighting=True, viewport_size=(512, 512))
