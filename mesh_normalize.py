import decorators
from mesh_data import MeshData
import pyfqmr
from trimesh import Trimesh
import os
import pyvista as pv


@decorators.time_func
@decorators.cache_result
def remesh_all_meshes(meshes: list[MeshData]) -> list[MeshData]:
    # load the mesh of every .obj file
    for mesh in meshes:
        mesh.trimesh_data = mesh.trimesh_data.process(validate=True)
        mesh.trimesh_data.fill_holes()

        pv_mesh = pv.wrap(mesh.trimesh_data)
        to_probe = pv.create_grid(pv_mesh)
        result = pv_mesh.sample(to_probe)

        mesh.trimesh_data = Trimesh(
            result.points, result.faces.reshape(-1, 4)[:, 1:])

        if len(mesh.trimesh_data.faces) < 1000:
            print(mesh.filename, str(len(mesh.trimesh_data.faces)))
            while len(mesh.trimesh_data.faces) < 1000:
                mesh.trimesh_data = mesh.trimesh_data.subdivide()
                print(str(len(mesh.trimesh_data.faces)))
        if len(mesh.trimesh_data.faces) > 5000:
            mesh.trimesh_data = simplify_mesh(mesh.trimesh_data)

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
