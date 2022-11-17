from tqdm import tqdm
import query
import numpy as np
import mesh_data
from mesh_data import MeshData, get_standard_feature_vec
import matplotlib.pyplot as plt
import os


def graph_ROC(meshes: list[MeshData]):

    knn_structures = [query.create_knn_structure(meshes, i) for i in range(2, len(meshes) + 1)]
    data_mean, data_std = mesh_data.get_mean_std(mesh_data.get_database_as_feature_matrix(meshes))
    meshes_subset = meshes

    true_positives = []
    true_negatives = []
    for mesh in tqdm(meshes_subset):
        feature_vector = get_standard_feature_vec(mesh, data_mean, data_std)
        feature_vector = feature_vector.reshape(1, -1)
        true_positives.append([])
        true_negatives.append([])
        for knn_structure in knn_structures:
            _, indices = knn_structure.kneighbors(feature_vector)
            indices = indices.reshape(-1)[1:]  # remove the first as that is the input shape
            tp = len([index for index in indices if meshes[index].mesh_class == mesh.mesh_class])
            tn = len(
                [
                    index
                    for index in range(len(meshes))
                    if index not in indices and meshes[index].mesh_class != mesh.mesh_class
                ]
            )
            true_positives[len(true_positives) - 1].append(tp)
            true_negatives[len(true_negatives) - 1].append(tn)
    true_positives = np.array(true_positives) / 19
    true_negatives = np.array(true_negatives) / len(meshes)

    os.makedirs(os.path.dirname("rocs/"), exist_ok=True)
    current_class = meshes_subset[0].mesh_class
    class_roc_sum = 0
    class_count = 0
    for mesh_i in range(len(meshes_subset)):
        if meshes_subset[mesh_i].mesh_class != current_class:
            plt.title(f"{current_class} AUROC {class_roc_sum/class_count:.2f}")
            plt.xlabel("TP")
            plt.ylabel("TN")
            plt.savefig(f"rocs/roc_{current_class}.png")
            plt.clf()
            current_class = meshes_subset[mesh_i].mesh_class
            class_roc_sum = 0
            class_count = 0
        plt.plot(true_positives[mesh_i], true_negatives[mesh_i], "-")
        class_roc_sum += np.sum(true_positives[mesh_i]) / len(meshes)
        class_count += 1

    plt.title(f"{current_class} AUROC {class_roc_sum/class_count:.2f}")
    plt.xlabel("TP")
    plt.ylabel("TN")
    plt.savefig(f"rocs/roc_{current_class}.png")
    plt.clf()

    class_roc_sum = 0
    class_count = 0
    for mesh_i in range(len(meshes_subset)):
        plt.plot(true_positives[mesh_i], true_negatives[mesh_i], "-")
        class_roc_sum += np.sum(true_positives[mesh_i]) / len(meshes)
        class_count += 1

    plt.title(f"global AUROC {class_roc_sum/class_count:.2f}")
    plt.xlabel("TP")
    plt.ylabel("TN")
    plt.savefig(f"rocs/roc_global.png")
    plt.clf()

    global_true_positives = np.sum(true_positives, axis=0) / len(meshes_subset)
    global_true_negatives = np.sum(true_negatives, axis=0) / len(meshes_subset)
    roc = np.sum(global_true_positives) / len(meshes)

    plt.plot(global_true_positives, global_true_negatives, "-")
    plt.title(f"averaged AUROC {roc:.2f}")
    plt.xlabel("TP")
    plt.ylabel("TN")
    plt.savefig(f"rocs/roc_averaged.png")
    plt.clf()
