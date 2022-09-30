import mesh_data
from trimesh import Trimesh

def GetBaryCenter(mesh: Trimesh):
    print(mesh.center_mass)
    return mesh.center_mass

def NormalizeTranslation(mesh: Trimesh):
    baryCenter = GetBaryCenter(mesh)
    for vertex in mesh.vertices:
        vertex -= baryCenter

def NormalizeTranslations(meshes: list[mesh_data.MeshData]):
    for mesh in meshes:
        baryCenter = GetBaryCenter(mesh.trimesh_data)
        for vertex in mesh.trimesh_data.vertices:
            vertex -= baryCenter

def GetBoundingBoxBiggestAxis(boundingbox: list[float]):
    Dx = abs(boundingbox[3] - boundingbox[0])
    Dy = abs(boundingbox[4] - boundingbox[1])
    Dz = abs(boundingbox[5] - boundingbox[2])

    return max(Dx, Dy, Dz)

def NormalizeScale(meshes: list[mesh_data.MeshData]):
    for mesh in meshes:
        scale_factor = 1 / GetBoundingBoxBiggestAxis(mesh.bounding_box)
        for vertex in mesh.trimesh_data.vertices:
            vertex *= scale_factor