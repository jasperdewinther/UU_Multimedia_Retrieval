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
import normalization


if __name__ == "__main__":
    # Find all files
    meshes = mesh_io.get_all_obj_files("./assets/")

    # Reduce dataset for faster computation
    meshes = meshes[500:700]

    # Load all meshes into ram
    meshes = mesh_io.get_all_meshes(meshes)

    meshes = descriptors.get_global_descriptors(meshes)
    mesh_data.summarize_data(
        meshes, "before_histograms.png", "before_data.csv")

    # Remesh all meshes, change the number of faces to fit in range
    meshes = mesh_normalize.remesh_all_meshes(meshes, 1000, 5000)

    # Normalize the location
    meshes = normalization.NormalizeTranslations(meshes)

    # Normalize the object scale
    meshes = normalization.NormalizeScales(meshes)

    # Normalize object alignment
    #meshes = normalization.NormalizeAlignments(meshes)

    # Calculate global descriptor
    meshes = descriptors.get_global_descriptors(meshes)

    # mesh_data.render_histogram(
    #    meshes, 100, 'broken_faces_count', 'broken_faces_count_hist_before.png')

    # Remove meshes which contain nan or inf values and throw away a portion of the dataset with the highest ratio of broken faces
    meshes = filter_io.remove_degenerate_models(meshes, 0.9)

    # Create histograms and database csv
    mesh_data.summarize_data(meshes, "after_histograms.png", "after_data.csv")

    # Select meshes to render
    torender = meshes

    # Render selected meshes
    renderer.render_meshes(torender)

    window = initGUI()

   # while True:                             # The Event Loop
    #    if not HandleGUIEvents(window):
    #        break

    # window.close()
