import tkinter
from board import Board
import numpy as np

# number of pixels in the cell (height = width)
cell_size = 15

# number of cells in the grid (height = width)
grid_size = 28

# add borders to the drawn window
border = 10

# window size
window_size = grid_size * cell_size

# leave border for functional buttons (like erase, exit)
buttons_border = 50

# determine where the bottom left cell begins
northwest_point = [border, border]

# choose between painting mode and erasing mode
erasing_mode = False


def paint_number(function_ref):
    def on_click(event):
        paint_and_guess(event)

    def on_clicked_draw(event):
        paint_and_guess(event)

    def paint_and_guess(event):
        global erasing_mode
        coordinates_x = (event.x - northwest_point[0]) / cell_size
        coordinates_y = (event.y - northwest_point[1]) / cell_size
        print('{:.2f} : {:.2f}'.format(coordinates_x, coordinates_y))
        if erasing_mode is True:
            board.remove_point(coordinates_x, coordinates_y)
        else:
            board.add_point(coordinates_x, coordinates_y)
        print(function_ref(np.asarray(board.get_image())))

    def clear_call_back():
        board.clear()

    def erase_call_back():
        global erasing_mode
        erasing_mode = True

    def draw_call_back():
        global erasing_mode
        erasing_mode = False

    root = tkinter.Tk()
    root.geometry('{}x{}'.format(window_size + border * 2, window_size + border * 2 + buttons_border))
    root.resizable(width=False, height=False)

    canvas = tkinter.Canvas(root, bg='white')
    canvas.bind('<B1-Motion> ', on_clicked_draw)
    canvas.bind('<Button-1>', on_click)
    canvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    button_canvas = tkinter.Canvas(root, bg='black')
    button_canvas.bind('<B1-Motion> ', on_clicked_draw)
    button_canvas.bind('<Button-1>', on_click)
    button_canvas.pack()
    button_canvas.config(width=window_size + border * 2, height=buttons_border)

    button_clear = tkinter.Button(button_canvas, text="Clear", command=clear_call_back)
    button_clear.grid(row=0, column=0, columnspan=1, sticky='w')

    button_paint = tkinter.Button(button_canvas, text="Draw", command=draw_call_back)
    button_paint.grid(row=0, column=1, columnspan=1, sticky='w')

    button_erase = tkinter.Button(button_canvas, text="Erase", command=erase_call_back)
    button_erase.grid(row=0, column=2, columnspan=1, sticky='w')

    board = Board(grid_size, cell_size, canvas)
    board.draw_grid()

    root.mainloop()
