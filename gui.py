import PySimpleGUI as sg
import mesh_io


def initGUI() -> sg.Window:
    sg.theme('DarkAmber')   # Add a touch of color
    # All the stuff inside your window.
    layout = [[sg.Text('Filepath')],
              [sg.Input(), sg.FileBrowse()],
              [sg.Button('Ok'), sg.Button('Cancel')]]

    # Create the Window
    window = sg.Window('MMR', layout)
    return window


def HandleGUIEvents(window: sg.Window) -> bool:
    event, input = window.read()
    print(event, input)
    model = mesh_io.get_mesh(input)
    if event == sg.WIN_CLOSED or event == 'Exit':
        return False

    return True
