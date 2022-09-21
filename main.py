from numpy.core.fromnumeric import mean
from asyncio import Handle
from gui import HandleGUIEvents, initGUI
import mesh_io
import renderer
import filter_io
import PySimpleGUI as sg


mesh_files = mesh_io.get_all_obj_files("./assets/")
meshes = mesh_io.get_all_meshes(mesh_files)

filter_io.get_class(mesh_files)
filter_io.get_faces_vertices(mesh_files)
filter_io.get_face_type(mesh_files)
filter_io.get_bounding_box(mesh_files)

renderer.render_meshes([meshes[6]])

window = initGUI()

while True:                             # The Event Loop
    if not HandleGUIEvents(window):
        break


window.close()
