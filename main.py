import mesh_io
import random
import pyrender

mesh_files = mesh_io.get_all_obj_files("./assets/")
meshes = mesh_io.get_all_meshes(mesh_files)

fuze_trimesh = random.choice(meshes)
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)
