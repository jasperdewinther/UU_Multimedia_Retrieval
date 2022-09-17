import mesh_io
import renderer

mesh_files = mesh_io.get_all_obj_files("./assets/")
meshes = mesh_io.get_all_meshes(mesh_files)


renderer.render_meshes(meshes[5:20])
