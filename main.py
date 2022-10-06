from numpy.core.fromnumeric import mean
from asyncio import Handle
from gui import HandleGUIEvents, initGUI
import mesh_io
import renderer
import filter_io
import decorators
import mesh_normalize
import descriptors
import pandas as pd
import mesh_data


if __name__ == "__main__":
    meshes = mesh_io.get_all_obj_files("./assets/")  # sets filename_field
    meshes = meshes[600:800]
    meshes = mesh_io.get_all_meshes(meshes)  # sets trimesh_model field
    meshes = mesh_normalize.remesh_all_meshes(meshes)  # normalize mesh
    meshes = filter_io.remove_degenerate_models(meshes)
    meshes = filter_io.output_filter(meshes)  # determine details
    meshes = descriptors.get_global_descriptors(meshes)
    mesh_data.summarize_data(meshes)

    print(mesh_data.generate_histogram(meshes, 100, 'vertex_count'))
    mesh_data.render_histogram(
        meshes, 100, 'vertex_count', 'vertex_count_hist.png')

    torender = meshes
    for mesh in torender:
        print(mesh.filename, "vertex count:", str(
            mesh.vertex_count), "face count:", str(mesh.face_count))
    renderer.render_meshes(torender)

    window = initGUI()

   # while True:                             # The Event Loop
    #    if not HandleGUIEvents(window):
    #        break

    # window.close()
