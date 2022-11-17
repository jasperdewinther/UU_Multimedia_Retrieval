from asyncio import Handle

import matplotlib.pyplot as plt
import pandas as pd
from numpy.core.fromnumeric import mean
from sklearn import pipeline
from sklearn.manifold import TSNE

import decorators
import evaluation
import filter_io
import mesh_data
import mesh_io
import mesh_normalize
import normalization
import pipeline_stages
import query
import renderer
from gui import HandleGUIEvents, initGUI
from renderer import RenderMode

PARALLEL_FOR_LOOP = True


if __name__ == "__main__":
    # Find all files
    meshes = mesh_io.get_all_obj_files("./assets/")

    meshes = [
        mesh
        for mesh in meshes
        if not any(
            classname in mesh.filename
            for classname in ["animal", "building", "furniture", "household", "miscellaneous", "plant", "vehicle"]
        )
    ]

    # Reduce dataset for faster computation
    # meshes = meshes[:200]

    # Load all meshes into ram
    meshes = pipeline_stages.get_all_meshes(meshes)
    # meshes = pipeline_stages.get_global_descriptors(meshes, 1000)
    # meshes, _ = pipeline_stages.get_global_descriptors(meshes, 1, 1)

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
    # meshes = pipeline_stages.remove_models_with_holes(meshes, 0.9)
    # meshes = pipeline_stages.remove_models_with_too_many_faces(meshes, 0.95)

    meshes, minmax_data = pipeline_stages.get_global_descriptors(meshes, 100000, 10000)

    # Create histograms and database csv
    # mesh_data.summarize_data(meshes, "after_histograms.png", "after_data.csv")
    #mesh_data.render_class_histograms(meshes, "histograms_after/")

    # meshes[191] = normalization.NormalizeFlip(meshes[191])
    # torender = meshes[190:200]
    # #torender = [meshes[191]]
    # renderer.render_meshes(torender)
    # windor = initGUI()

    # max_index = mesh_data.get_outlier_high_mesh(meshes, "obb_volume")
    # median_index = mesh_data.get_median_mesh(meshes, "obb_volume")
    # min_index = mesh_data.get_outlier_low_mesh(meshes, "obb_volume")
    # print(max_index, median_index, min_index)
    # Select meshes to render
    # torender = [max_index, median_index, min_index]
    # print(torender)

    # Render selected meshes
    # renderer.render_meshes(torender, RenderMode.INORDER)

    # evaluation.graph_ROC(meshes)

    # exit()
    knn = query.create_knn_structure(meshes, 50)

    # sword = [mesh for mesh in meshes if "m702.obj" in mesh.filename][0]
    # pig = [mesh for mesh in meshes if "m100.obj" in mesh.filename][0]

    # nearest = query.query_knn(pig, meshes, knn, 50)
    # nearest = sorted(nearest, key=lambda x: x[1])
    # [print(mesh[1]) for mesh in nearest]
    # torender = [mesh[0] for mesh in nearest]
    # renderer.render_meshes(torender)
    window = initGUI()
    data_mean, data_std = mesh_data.get_mean_std(mesh_data.get_database_as_feature_matrix(meshes))

    # renderer.render_meshes(meshes, RenderMode.TSNE)

    while True:  # The Event Loop
        event_return = HandleGUIEvents(window, minmax_data)
        if isinstance(event_return, bool):
            if not event_return:
                break
        if isinstance(event_return, mesh_data.MeshData):
            nearest = query.query_knn(event_return, meshes, knn, data_mean, data_std)
            # nearest = query.query_brute_force(event_return, meshes, data_mean, data_std, 9)
            # [0] = mesh, [1] = distance, [2] individual dist per component
            #query.show_distances(nearest)
            print(nearest)

            distances = []
            for mesh in nearest:
                distances.append(mesh[len(nearest[0]) - 2])

            print(distances)
            nearest = sorted(nearest, key=lambda x: x[1])
            torender = [mesh[0] for mesh in nearest]
            viewer = renderer.render_meshes(torender, RenderMode.INORDER)

    viewer.close()
    window.close()
