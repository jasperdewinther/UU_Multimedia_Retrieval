import trimesh
import os
import csv

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
        #writer.writerow([vertices, faces])
    f.close()
    

def get_face_type():
    return

def get_bounding_box():
    return
