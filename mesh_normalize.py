import decorators
import mesh_data
import pyfqmr
import trimesh
import os


@decorators.time_func
@decorators.cache_result
def remesh_all_meshes(meshes: list[mesh_data.MeshData]) -> list[mesh_data.MeshData]:
    # load the mesh of every .obj file
    for mesh in meshes:
        if len(mesh.trimesh_data.vertices) < 5000:
            while len(mesh.trimesh_data.vertices) < 5000:
                mesh.trimesh_data = mesh.trimesh_data.subdivide()
        if len(mesh.trimesh_data.vertices) > 10_000:
            mesh.trimesh_data = simplify_mesh(mesh.trimesh_data)

    return meshes


def simplify_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(
        mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(
        target_count=7000, verbose=0)
    vertices, faces, normals = mesh_simplifier.getMesh()
    mesh.vertices = vertices
    mesh.faces = faces
    mesh.vertex_normals = normals
    new_mesh = trimesh.Trimesh(vertices)
    return new_mesh
