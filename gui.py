from typing import Union
import PySimpleGUI as sg
from descriptors import get_global_descriptor
from mesh_data import MeshData
import mesh_io
import trimesh
import descriptors

import mesh_normalize
import normalization


def initGUI() -> sg.Window:
    sg.theme("DarkAmber")  # Add a touch of color
    # All the stuff inside your window.
    layout = [[sg.Text("Filepath")], [sg.Input(), sg.FileBrowse()], [sg.Button("Ok"), sg.Button("Cancel")]]

    # Create the Window
    window = sg.Window("MMR", layout)
    return window


def HandleGUIEvents(window: sg.Window, minmax_data: list[float]) -> Union[MeshData, bool]:
    event, input = window.read()
    if input is not None and "Browse" in input:
        meshdata = PrepareInputModel(input["Browse"], minmax_data)
        return meshdata
    if event == sg.WIN_CLOSED or event == "Exit":
        return False

    return True


def PrepareInputModel(filename: str, minmax_data: list[float]) -> MeshData:
    data = MeshData()
    data.filename = filename
    data = mesh_io.set_trimesh(data)
    data = mesh_normalize.remesh(data, 1000, 5000)
    data = normalization.NormalizeTranslation(data)
    data = normalization.NormalizeScale(data)
    data = normalization.NormalizeAlignment(data)
    data = descriptors.get_global_descriptor(data, 5000, 5000)
    data = descriptors.gen_histograms(data, minmax_data)

    return data
