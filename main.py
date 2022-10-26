from numpy.core.fromnumeric import mean
from asyncio import Handle

from sklearn import pipeline
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
import query
import matplotlib.pyplot as plt
import pipeline_stages

PARALLEL_FOR_LOOP = False


if __name__ == "__main__":
    # Find all files
    meshes = mesh_io.get_all_obj_files("./assets/")

    # Reduce dataset for faster computation

    # Load all meshes into ram
    meshes = pipeline_stages.get_all_meshes(meshes)

    meshes = pipeline_stages.remove_nan_inf_models(meshes)

    # meshes = descriptors.get_global_descriptors(meshes, 1000)
    # mesh_data.render_class_histograms(meshes, "histograms/")
    # mesh_data.summarize_data(meshes, "before_histograms.png", "before_data.csv")

    # Remesh all meshes, change the number of faces to fit in range
    meshes = pipeline_stages.remesh_all_meshes(meshes, 1000, 5000)

    # Normalize the location
    meshes = pipeline_stages.NormalizeTranslations(meshes)

    # Normalize the object scale
    meshes = pipeline_stages.NormalizeScales(meshes)

    # Normalize object alignment
    # meshes = pipeline_stages.NormalizeAlignments(meshes)

    # Calculate global descriptor
    meshes = pipeline_stages.get_global_descriptors(meshes, 1000)
    meshes = pipeline_stages.get_shape_properties(meshes, 1000)

    # mesh_data.render_histogram(
    #    meshes, 100, 'broken_faces_count', 'broken_faces_count_hist_before.png')

    # Remove meshes which contain nan or inf values and throw away a portion of the dataset with the highest ratio of broken faces
    meshes = pipeline_stages.remove_models_with_holes(meshes, 0.9)
    meshes = pipeline_stages.remove_models_with_too_many_faces(meshes, 0.95)

    # Create histograms and database csv
    mesh_data.summarize_data(meshes, "after_histograms.png", "after_data.csv")
    # mesh_data.render_class_histograms(meshes, "histograms_after/")

    # knn = query.create_knn_structure(meshes, 5)

    # nearest = query.query_knn(meshes[654], meshes, knn, 5)
    # print(nearest)

    # Select meshes to render
    torender = [
        meshes[mesh_data.get_outlier_high_mesh(meshes, "broken_faces_count")],
        meshes[mesh_data.get_median_mesh(meshes, "broken_faces_count")],
        meshes[mesh_data.get_outlier_low_mesh(meshes, "broken_faces_count")],
    ]
    print(torender)

    # Render selected meshes
    renderer.render_meshes(torender)

    window = initGUI()

# while True:                             # The Event Loop
#    if not HandleGUIEvents(window):
#        break

# window.close()
