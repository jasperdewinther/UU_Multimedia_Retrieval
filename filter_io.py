import trimesh
import os
import csv
import numpy as np

def output_filter(mesh_files):
    # creates csv file with data from every shape
    f = open(os.getcwd() + '/faces_vertices.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['name', 'class', '# faces', '# vertices', 'triangles', 'quads', 'bounding box'])
    for meshes in mesh_files:
        class_shape = get_class(meshes)
        faces, vertices = get_faces_vertices(meshes)
        triangles, quads = get_face_type(meshes)
        bounding_box = get_bounding_box(meshes)
        writer.writerow([os.path.basename(meshes), class_shape, faces, vertices, triangles, quads, bounding_box])
    f.close()

def get_class(meshes):
    # finds the the class of the shape (for sheep, etc this is "assets")        
    full_path = os.path.dirname(meshes)
    class_shape = os.path.basename(full_path)
    return class_shape

def get_faces_vertices(meshes):
    # finds number of vertices and faces of the shape and writes to csv file
    mesh = trimesh.load_mesh(meshes)
    faces = mesh.faces.shape[0]
    vertices = mesh.vertices.shape[0]
    return faces, vertices
    
def get_face_type(meshes):
    # find the types of faces
    triangles = False
    quads = False
    f = open(meshes, 'r')
    for lines in f:
        lines_array = lines.split()
        if len(lines_array) > 0:
            if lines_array[0] == 'f':
                if len(lines_array) == 4: triangles = True
                elif len(lines_array) == 5: quads = True
            if triangles == True and quads == True:
                return triangles, quads
    return triangles, quads

def get_bounding_box(meshes):
    # find bounding box of the shape
    bounding_box = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf] #[x_min, y_min, z_min, x_max, y_max, z_max]
    f = open(meshes, 'r')
    for lines in f:
        lines_array = lines.split()
        if len(lines_array) > 0:
            if lines_array[0] == 'v':
                vertices = []
                vertices.append(float(lines_array[1]))
                vertices.append(float(lines_array[2]))
                vertices.append(float(lines_array[3]))
                if bounding_box[0] > vertices[0]: bounding_box[0] = vertices[0]
                if bounding_box[1] > vertices[1]: bounding_box[1] = vertices[1]
                if bounding_box[2] > vertices[2]: bounding_box[2] = vertices[2]
                if bounding_box[3] < vertices[0]: bounding_box[3] = vertices[0]
                if bounding_box[4] < vertices[1]: bounding_box[4] = vertices[1]
                if bounding_box[5] < vertices[2]: bounding_box[5] = vertices[2]
    return bounding_box
