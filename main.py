from numpy.core.fromnumeric import mean
import mesh_io
import renderer
import filter_io

mesh_files = mesh_io.get_all_obj_files("./assets/")
meshes = mesh_io.get_all_meshes(mesh_files)

filter_io.get_class(mesh_files)
filter_io.get_faces_vertices(mesh_files)
filter_io.get_face_type(mesh_files)
filter_io.get_bounding_box(mesh_files)

renderer.render_meshes(meshes[5:20])
