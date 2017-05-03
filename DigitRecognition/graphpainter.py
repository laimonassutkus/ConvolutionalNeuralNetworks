import tkinter

# number of pixels in the cell (height = width)
cell_size = 15

# number of cells in the grid (height = width)
grid_size = 20

# add borders to the drawn window
border = 10

# window size
window_size = grid_size * cell_size

# determine where the bottom left cell begins
southwest_point = [border, border]


def draw_rect(coordinates_x, coordinates_y):
    canvas.create_rectangle(
        int(coordinates_x) * cell_size + border,
        int(coordinates_y) * cell_size + border,
        int(coordinates_x) * cell_size + cell_size + border,
        int(coordinates_y) * cell_size + cell_size + border, fill='black', outline='black')


def on_click(event):
    coordinates_x = (event.x - southwest_point[0]) / cell_size
    coordinates_y = (event.y - southwest_point[1]) / cell_size
    print('{:.2f} : {:.2f}'.format(coordinates_x, coordinates_y))
    draw_rect(coordinates_x, coordinates_y)


def on_clicked_draw(event):
    coordinates_x = (event.x - southwest_point[0]) / cell_size
    coordinates_y = (event.y - southwest_point[1]) / cell_size
    draw_rect(coordinates_x, coordinates_y)

root = tkinter.Tk()
root.geometry('{}x{}'.format(window_size * 2 + border, window_size + border * 2))
root.resizable(width=False, height=False)

canvas = tkinter.Canvas(root, bg='white')
canvas.bind('<B1-Motion> ', on_clicked_draw)
canvas.bind('<Button-1>', on_click)
canvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


for dist in range(0, grid_size + 1):
    canvas.create_line(0 + border, cell_size * dist + border, window_size + border, cell_size * dist + border)
    canvas.create_line(cell_size * dist + border, 0 + border, cell_size * dist + border, window_size + border)

root.mainloop()

















# import matplotlib
# matplotlib.use("Qt4Agg")
#
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# # number of pixels in the cell (height = width) it is a hard-coded value
# cell_size = 22.5
#
# # number of cells in the grid (height = width)
# grid_size = 20
#
# # do not change a hard-coded value
# WS_start_point = [132, 94]
#
# grid = np.zeros(shape=(grid_size, grid_size))
#
# fig = plt.figure()
# plt.title('Placeholder')
#
# start_coordinates = -1
# for dist in range(0, grid_size + 1):
#     start_coordinates += 1
#     plt.plot([start_coordinates, start_coordinates], [0, grid_size], color=[0.75, 0.75, 0.75], linewidth=0.5)
#     plt.plot([0, grid_size], [start_coordinates, start_coordinates], color=[0.75, 0.75, 0.75], linewidth=0.5)
#
# is_pressed = False
#
#
# def on_click(event):
#     global is_pressed
#     is_pressed = True
#    #@ grid[]
#     print('{:.2f} : {:.2f}'
#           .format((event.x - WS_start_point[0]) / cell_size, (event.y - WS_start_point[1]) / cell_size))
#
#
# def on_release(event):
#     global is_pressed
#     is_pressed = False
#
#
# def on_draw(event):
#     if is_pressed:
#         print('{:.2f} : {:.2f}'.format(event.x, event.y))
#
# plt.axis('equal')
# fig.canvas.mpl_connect('button_press_event', on_click)
# fig.canvas.mpl_connect('button_release_event', on_release)
# fig.canvas.mpl_connect('motion_notify_event', on_draw)
# win = fig.canvas.window()
# win.setFixedSize(700, 700)
# plt.show()
