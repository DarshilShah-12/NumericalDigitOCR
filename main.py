from tkinter import *
import PIL
from PIL import Image, ImageDraw
import numpy as np
import hl_1_neural_network

def submit():
    global default
    resized_img = image1.resize((28,28)).convert("L")
    img_arr = np.asarray(resized_img)
    new_img_arr = np.zeros((28,28))

    for i in range(len(img_arr)):
        for j in range(len(img_arr[i])):
            new_img_arr[i][j] = 255 - img_arr[i][j]

    neural_net = hl_1_neural_network.NeuralNetwork()
    neural_net.open_load()
    neural_net.feed_forward(new_img_arr)
    default.pack_forget()
    default = Message(win, text=("Result:", np.argmax(neural_net.ao)))
    default.pack()
    return

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y

def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_rectangle((lastx, lasty, x, y), width=20)
    draw.line((lastx, lasty, x, y), fill='black', width=10)
    lastx, lasty = x, y
def clear():
    global image1
    global draw
    global default
    cv.delete('all')
    image1 = PIL.Image.new('RGB', (280, 280), 'white')
    draw = ImageDraw.Draw(image1)
    default.pack_forget()
    default = Message(win, text="Result:")
    default.pack()
def exit_mainloop():
    exit()

win = Tk()
win.resizable(False, False)
win.title("Paint - made in Python")
lastx, lasty = None, None

cv = Canvas(win, width=280, height=280, bg='white')
image1 = PIL.Image.new('RGB', (280, 280), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

reset=Button(text='Reset canvas',command=clear)
reset.pack(side=LEFT)

submit_ = Button(text="Submit digit", command=submit)
submit_.pack(side=LEFT)

_exit=Button(text='Exit', command=exit_mainloop)
_exit.pack(side=LEFT)

default = Message(win, text="Result:")
default.pack()

win.mainloop()