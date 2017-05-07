from tkinter import *
nn_stats = ["aa", "bb", "cc"]
root = Tk()
t = Text(root)
t.insert(INSERT, "Hello World!\nHetttttt")
t.pack()
t.delete('1.0', END)
t.insert(INSERT, "Simple neural network:\n" + ''.join(str(x) for x in nn_stats))
root.mainloop()