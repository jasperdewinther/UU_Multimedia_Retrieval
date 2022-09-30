import mesh_data

def GetBaryCenter(mesh: mesh_data.MeshData):
    print(mesh.trimesh_data.center_mass)
    return mesh.center_mass

def NormalizeTranslations(meshes: list[mesh_data.MeshData]):
    baryCenter = GetBaryCenter
    for mesh in meshes:
        for vertex in mesh.trimesh_data.vertices:
            vertex -= baryCenter

def GetBoundingBoxBiggestAxis(boundingbox: list[float]):
    Dx = boundingbox[0] - boundingbox[3]
    Dy = boundingbox[1] - boundingbox[4]
    Dz = boundingbox[2] - boundingbox[5]

    return max(Dx, Dy, Dz)

def NormalizeScale(meshes: list[mesh_data.MeshData]):
    for mesh in meshes:
        scale_factor = 1 / GetBoundingBoxBiggestAxis(mesh.bounding_box)
        for vertex in mesh.trimesh_data.vertices:
            vertex *= scale_factor