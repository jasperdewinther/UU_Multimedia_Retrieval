import trimesh
import mesh_data
import math
import ast
import pandas as pd
import os
import decorators


@decorators.time_func
@decorators.cache_result
def get_global_descriptors(meshes: list[mesh_data.MeshData]) -> list[mesh_data.MeshData]:
    # finds the surface area, compactness, axis-aligned boudning-box volume, diameter and eccentricity
    counter = 0
    for mesh in meshes:
        # surface area
        surface_area = mesh.trimesh_data.area

        # compactness
        volume = mesh.trimesh_data.volume
        compactness = (surface_area**3) / (36*math.pi*(volume**2))

        # axis-aligned bounding-box volume
        data = pd.read_csv(os.getcwd() + '/faces_vertices.csv', sep=',')
        bounding_box = data['bounding box'][counter]
        boxList = ast.literal_eval(bounding_box)
        x1 = float(boxList[0])
        y1 = float(boxList[1])
        z1 = float(boxList[2])
        x2 = float(boxList[3])
        y2 = float(boxList[4])
        z2 = float(boxList[5])
        bb_volume = abs(x2 - x1) * abs(y2 - y1) * abs(z2 - z1)

        # diameter
        diameter = max(x2 - x1, y2 - y1, z2 - z1)

        mesh.surface_area = surface_area
        mesh.compactness = compactness
        mesh.bb_volume = bb_volume
        mesh.diameter = diameter
        # eccentricity
        # print(str(surface_area) + ", " + str(compactness) +
        #      ", " + str(bb_volume) + ", " + str(diameter))
    return meshes
