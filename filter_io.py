from trimesh import Trimesh
import os
import csv
import numpy as np
import decorators
import mesh_data


@decorators.time_func
@decorators.cache_result
def output_filter(meshes: list[mesh_data.MeshData]) -> list[mesh_data.MeshData]:
    # creates csv file with data from every shape
    f = open(os.getcwd() + '/faces_vertices.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['name', 'class', '# faces', '# vertices',
                    'triangles', 'quads', 'bounding box'])
    for mesh in meshes:
        class_shape = get_class(mesh.filename)
        faces, vertices = get_faces_vertices(mesh.trimesh_data)
        triangles, quads = get_face_type(mesh.filename)
        bounding_box = get_bounding_box(mesh.filename)
        writer.writerow([os.path.basename(mesh.filename), class_shape,
                        faces, vertices, triangles, quads, bounding_box])

        mesh.vertex_count = triangles + quads
        mesh.bounding_box = bounding_box
    f.close()
    return meshes


def get_class(mesh_file: str):
    # finds the the class of the shape (for sheep, etc this is "assets")
    full_path = os.path.dirname(mesh_file)
    class_shape = os.path.basename(full_path)
    return class_shape


def get_faces_vertices(mesh: Trimesh):
    # finds number of vertices and faces of the shape and writes to csv file
    faces = mesh.faces.shape[0]
    vertices = mesh.vertices.shape[0]
    return faces, vertices


def get_face_type(mesh_file: str):
    # find the types of faces
    triangles = False
    quads = False
    f = open(mesh_file, 'r')
    for lines in f:
        lines_array = lines.split()
        if len(lines_array) > 0:
            if lines_array[0] == 'f':
                if len(lines_array) == 4:
                    triangles = True
                elif len(lines_array) == 5:
                    quads = True
            if triangles == True and quads == True:
                return triangles, quads
    return triangles, quads


def get_bounding_box(mesh_file: str):
    # find bounding box of the shape
    # [x_min, y_min, z_min, x_max, y_max, z_max]
    bounding_box = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]
    f = open(mesh_file, 'r')
    for lines in f:
        lines_array = lines.split()
        if len(lines_array) > 0:
            if lines_array[0] == 'v':
                vertices = []
                vertices.append(float(lines_array[1]))
                vertices.append(float(lines_array[2]))
                vertices.append(float(lines_array[3]))
                if bounding_box[0] > vertices[0]:
                    bounding_box[0] = vertices[0]
                if bounding_box[1] > vertices[1]:
                    bounding_box[1] = vertices[1]
                if bounding_box[2] > vertices[2]:
                    bounding_box[2] = vertices[2]
                if bounding_box[3] < vertices[0]:
                    bounding_box[3] = vertices[0]
                if bounding_box[4] < vertices[1]:
                    bounding_box[4] = vertices[1]
                if bounding_box[5] < vertices[2]:
                    bounding_box[5] = vertices[2]
    return bounding_box