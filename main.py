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

PARALLEL_FOR_LOOP = True


if __name__ == "__main__":
    # Find all files
    meshes = mesh_io.get_all_obj_files("./assets/")

    # Reduce dataset for faster computation
    # meshes = meshes[:200]

    # Load all meshes into ram
    meshes = pipeline_stages.get_all_meshes(meshes)
    # meshes = pipeline_stages.get_global_descriptors(meshes, 1000)

    # max_index = mesh_data.get_outlier_high_mesh(meshes, "broken_faces_count")
    # median_index = mesh_data.get_median_mesh(meshes, "broken_faces_count")
    # min_index = mesh_data.get_outlier_low_mesh(meshes, "broken_faces_count")
    # print(max_index, median_index, min_index)
    ## Select meshes to render
    # torender = [
    #    max_index,
    # ]
    # print(torender)
    #
    ## Render selected meshes
    # renderer.render_meshes(torender)

    meshes = pipeline_stages.remove_nan_inf_models(meshes)

    # Remesh all meshes, change the number of faces to fit in range
    meshes = pipeline_stages.remesh_all_meshes(meshes, 1000, 5000)

    # Normalize the location
    meshes = pipeline_stages.NormalizeTranslations(meshes)

    # Normalize the object scale
    meshes = pipeline_stages.NormalizeScales(meshes)

    # Normalize object alignment
    meshes = pipeline_stages.NormalizeAlignments(meshes)

    # Flipping objects
    meshes = pipeline_stages.NormalizeFlips(meshes)

    # Calculate global descriptor
    meshes, _ = pipeline_stages.get_global_descriptors(meshes, 1, 1)
    # meshes = pipeline_stages.get_shape_properties(meshes, 5000)

    # mesh_data.render_histogram(
    #    meshes, 100, 'broken_faces_count', 'broken_faces_count_hist_before.png')

    # Remove meshes which contain nan or inf values and throw away a portion of the dataset with the highest ratio of broken faces
    meshes = pipeline_stages.remove_models_with_holes(meshes, 0.9)
    meshes = pipeline_stages.remove_models_with_too_many_faces(meshes, 0.95)

    meshes, minmax_data = pipeline_stages.get_global_descriptors(meshes, 5000, 5000)

    # Create histograms and database csv
    # mesh_data.summarize_data(meshes, "after_histograms.png", "after_data.csv")
    # mesh_data.render_class_histograms(meshes, "histograms_after/")

    # meshes[191] = normalization.NormalizeFlip(meshes[191])
    # torender = meshes[190:200]
    # #torender = [meshes[191]]
    # renderer.render_meshes(torender)
    # windor = initGUI()

    knn = query.create_knn_structure(meshes, 9)

    # sword = [mesh for mesh in meshes if "m702.obj" in mesh.filename][0]
    # pig = [mesh for mesh in meshes if "m100.obj" in mesh.filename][0]

    # nearest = query.query_knn(pig, meshes, knn, 50)
    # nearest = sorted(nearest, key=lambda x: x[1])
    # [print(mesh[1]) for mesh in nearest]
    # torender = [mesh[0] for mesh in nearest]
    # renderer.render_meshes(torender)
    window = initGUI()

    while True:  # The Event Loop
        event_return = HandleGUIEvents(window, minmax_data)
        if isinstance(event_return, bool):
            if not event_return:
                break
        if isinstance(event_return, mesh_data.MeshData):
            nearest = query.query_knn(event_return, meshes, knn, 9)
            # [0] = mesh, [1] = distance, [2] individual dist per component
            query.show_distances(nearest)
            nearest = sorted(nearest, key=lambda x: x[1])
            torender = [mesh[0] for mesh in nearest]
            viewer = renderer.render_meshes(torender)

    viewer.close()
    window.close()
