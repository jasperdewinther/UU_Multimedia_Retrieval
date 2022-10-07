import decorators
from mesh_data import MeshData
import pyfqmr
from trimesh import Trimesh
import os
import pyvista as pv
import pymeshfix
from tqdm import tqdm
import pymeshfix as mf


@decorators.time_func
@decorators.cache_result
def remesh_all_meshes(meshes: list[MeshData]) -> list[MeshData]:
    # load the mesh of every .obj file
    for mesh in meshes:
        # print(
        #    f"vertices {len(mesh.trimesh_data.vertices)} faces {len(mesh.trimesh_data.faces)}")
        #pv_mesh = pv.wrap(mesh.trimesh_data)
        #meshfix = mf.MeshFix(pv_mesh)
        # meshfix.repair()
        # to_probe = pv.create_grid(pv_mesh)
        # result = pv_mesh.sample(to_probe)
        # print(f"{mesh.filename}")
        # print("pymeshfixing")
        # vclean, fclean = pymeshfix.clean_from_arrays(
        #    mesh.trimesh_data.vertices, mesh.trimesh_data.faces)
        # mesh.trimesh_data = Trimesh(vclean, fclean)
        # print(
        #    f"vertices {len(mesh.trimesh_data.vertices)} faces {len(mesh.trimesh_data.faces)}")
        # print("pymeshfixed")
        #pv_mesh = meshfix.mesh
        # mesh.trimesh_data = Trimesh(
        #    pv_mesh.points, pv_mesh.faces.reshape(-1, 4)[:, 1:])

        # print(
        #    f"vertices {len(mesh.trimesh_data.vertices)} faces {len(mesh.trimesh_data.faces)}")

        mesh.trimesh_data = mesh.trimesh_data.process(validate=True)
        mesh.trimesh_data.fill_holes()

        if len(mesh.trimesh_data.faces) < 1000:
            if len(mesh.trimesh_data.faces) == 0:
                print("0 detected")
            while len(mesh.trimesh_data.faces) < 1000:
                print(f"adding faces {len(mesh.trimesh_data.faces)}")
                mesh.trimesh_data = mesh.trimesh_data.subdivide()
        if len(mesh.trimesh_data.faces) > 5000:
            print(f"reducing faces {len(mesh.trimesh_data.faces)}")
            mesh.trimesh_data = simplify_mesh(mesh.trimesh_data)

        # pv_mesh = pv.wrap(mesh.trimesh_data)
        # pv_mesh = mf.MeshFix(pv_mesh)
        # mesh.trimesh_data = Trimesh(pv_mesh.points(), pv_mesh.faces())

    return meshes


def simplify_mesh(mesh: Trimesh) -> Trimesh:
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(
        mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(
        target_count=5000, verbose=0)
    vertices, faces, normals = mesh_simplifier.getMesh()
    new_mesh = Trimesh(vertices, faces, normals)
    return new_mesh
