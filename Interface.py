from graphics import *
import numpy as np
# from neural_network import *

grid_size = 28
radius = 10
pos = 2*radius

grid = []
digit = ""

win = GraphWin("Natural Digit Reader", (grid_size + 1)*pos, (grid_size + 3)*pos)
submitted = False

def main():
    global submitted, grid, win, digit

    win.bind('<B1-Motion>', drag)

    submit_button = Rectangle(Point(0.375*(grid_size + 1)*pos, (grid_size + 1)*pos), Point(0.625*(grid_size + 1)*pos, (grid_size+2.5)*pos))
    submit_text = Text(Point(0.5*(grid_size + 1)*pos, (grid_size + 1.75)*pos), "Submit")
    submit_button.draw(win)
    submit_text.draw(win)

    refresh_button = Rectangle(Point(pos, (grid_size + 1)*pos), Point(0.225*(grid_size + 1)*pos, (grid_size+2.5)*pos))
    refresh_text = Text(Point(0.125*(grid_size + 1)*pos, (grid_size + 1.75)*pos), "Refresh")
    refresh_button.draw(win)
    refresh_text.draw(win)

    digit_intro = Text(Point(0.8*(grid_size + 1)*pos, (grid_size + 1.75)*pos), "Number:")
    digit_text = Text(Point(0.875*(grid_size + 1)*pos, (grid_size + 1.75)*pos), digit)
    digit_text.setSize(18)
    digit_intro.draw(win)
    
    for i in range(grid_size):
        grid.append([])
        for j in range(grid_size):
            grid[i].append(Node((j+1)*pos, (i+1)*pos, radius))


    for i in range(len(grid)):
        for j in range(len(grid[i])):
            grid[i][j].draw(win)


    while True:
        click = win.getMouse()
        x = click.getX()
        y = click.getY()

        if((x >= submit_button.getP1().getX() and x <= submit_button.getP2().getX()) and (y >= submit_button.getP1().getY() and y <= submit_button.getP2().getY())):
            digit_text.setText(submit(grid))
            digit_text.draw(win)
            submitted = True
        
        elif((x >= refresh_button.getP1().getX() and x <= refresh_button.getP2().getX()) and (y >= refresh_button.getP1().getY() and y <= refresh_button.getP2().getY())):
            clear(grid)
            digit_text.undraw()
            submitted = False

        elif(not submitted):
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if(grid[i][j].inside(x,y)):
                        grid[i][j].changeColour()
                

class Node(Circle):

    def __init__(self, x, y, radius, filled = False):
        Circle.__init__(self, Point(x, y), radius)
        self.x = x
        self.y = y
        self.filled = filled
    
    def getX(self):
        return self.x

    def getY(self):
        return self.y
    
    def changeColour(self):
        if(self.filled):
            self.setFill(color="")
        else:
            self.setFill(color="black")
        
        self.filled = not self.filled

    def fill(self):
        if(not self.filled):
            self.setFill(color="black")
            self.filled = not self.filled

    def clear(self):
        if(self.filled):
            self.setFill(color="")
            self.filled = not self.filled

    def isFilled(self):
        return self.filled

    def inside(self, x, y):
        return (x >= (self.getX() - self.getRadius()) and x <= (self.getX() + self.getRadius())) and (y >= (self.getY() - self.getRadius()) and y <= (self.getY() + self.getRadius()))
        

def submit(grid):
    global digit

    values = []

    for i in range(len(grid)):
        values.append([])
        for j in range(len(grid)):
            if(grid[i][j].isFilled()):
                values[i].append(1)
            else:
                values[i].append(0)

    np_values = np.array(values)

    # print (np_values)

    return str(output(np_values))


def clear(grid):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            grid[i][j].clear()
            grid[i][j].undraw()
            grid[i][j].draw(win)
    
    # print('Grid Cleared')


def drag(event):
    global submitted, grid

    if(not submitted):
        x = event.x
        y = event.y
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if(grid[i][j].inside(x,y)):
                    grid[i][j].fill()
                    break
            else:
                continue
            break


main()