import trimesh
import pyrender


fuze_trimesh = trimesh.load('assets/sheep.obj')
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)

# add recursive obj file finder
# add multiple model viewer (instead of only showing one), to see top n results
# add buttons to gui
