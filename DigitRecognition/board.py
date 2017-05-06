class Board:
    board = None
    canvas = None
    grid_size = -1
    cell_size = -1
    window_size = -1
    border = 10

    def __init__(self, grid_size, cell_size, canvas):
        self.board = [[0] * grid_size for _ in range(grid_size)]
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.canvas = canvas
        self.window_size = grid_size * cell_size
        pass

    def add_point(self, x, y):
        if int(x) < self.grid_size and int(y) < self.grid_size and int(x) >= 0 and int(y) >= 0:
            start_x = int(x) * self.cell_size + self.border
            start_y = int(y) * self.cell_size + self.border
            self.canvas.create_rectangle(
                start_x,
                start_y,
                int(x) * self.cell_size + self.cell_size + self.border,
                int(y) * self.cell_size + self.cell_size + self.border, fill='black', outline='black')
            # swap rows to x and columns to y
            self.board[int(y)][int(x)] = 255

    def remove_point(self, x, y):
        if int(x) < self.grid_size and int(y) < self.grid_size and int(x) >= 0 and int(y) >= 0:
            start_x = int(x) * self.cell_size + self.border
            start_y = int(y) * self.cell_size + self.border
            self.canvas.create_rectangle(
                start_x,
                start_y,
                int(x) * self.cell_size + self.cell_size + self.border,
                int(y) * self.cell_size + self.cell_size + self.border, fill='white', outline='black')
            # swap rows to x and columns to y
            self.board[int(y)][int(x)] = 0

    def draw_grid(self):
        for dist in range(0, self.grid_size + 1):
            self.canvas.create_line(0 + self.border, self.cell_size * dist + self.border,
                                    self.window_size + self.border, self.cell_size * dist + self.border)
            self.canvas.create_line(self.cell_size * dist + self.border, 0 + self.border,
                                    self.cell_size * dist + self.border, self.window_size + self.border)

    def get_image(self):
        return self.board

    def clear(self):
        for index_x in range(0, self.grid_size):
            for index_y in range(0, self.grid_size):
                self.remove_point(index_x, index_y)
        return self.board
