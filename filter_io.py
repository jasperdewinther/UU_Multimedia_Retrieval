import trimesh
import os
import csv
import numpy as np

def get_class(mesh_files):
    # finds the the class of the shape (for sheep, etc this is "assets")
    for meshes in mesh_files:
        full_path = os.path.dirname(meshes)
        class_shape = os.path.basename(full_path)

def get_faces_vertices(mesh_files):
    # finds number of vertices and faces of the shape and writes to csv file
    f = open(os.getcwd() + '/faces_vertices.csv', 'w')
    writer = csv.writer(f)
    #writer.writerow(['vertices', 'faces'])
    for meshes in mesh_files:
        mesh = trimesh.load_mesh(meshes)
        vertices = mesh.vertices.shape[0]
        faces = mesh.faces.shape[0]
        trimesh.Scene().bounds
        #writer.writerow([vertices, faces])
    f.close()
    
def get_face_type(mesh_files):
    # find the types of faces
    for meshes in mesh_files:
        triangles = False
        quads = False
        f = open(meshes, 'r')
        for lines in f:
            lines_array = lines.split()
            if len(lines_array) > 0:
                if lines_array[0] == 'f':
                    if len(lines_array) == 4: triangles = True
                    elif len(lines_array) == 5: quads = True
                if triangles == True and quads == True: break

def get_bounding_box(mesh_files):
    # find bounding box of the shape
    for meshes in mesh_files:
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
        f.close()
