import trimesh


class MeshData:
    filename: str
    trimesh_data: trimesh.Trimesh

    def __init__(self):
        self.filename = None
        self.trimesh_data = None
        self.bounding_box = None
        self.vertex_count = None
