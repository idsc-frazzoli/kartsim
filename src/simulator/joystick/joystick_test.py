#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 22.05.19 20:42

@author: mvb
"""
import pygame
import time

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

app = QtGui.QApplication([])
#
win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1600, 800)
win.setWindowTitle('steering test')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p1 = win.addPlot(title="axes")#, setAspectLocked=True)
#    p1.plot(x,y, pen=(255,255,255), name="Red curve")
p1.showGrid(x=True, y=True)
# p1.setAspectLocked(lock=True, ratio=1)




#

runSimulation = True
#    first = True
axes_list = np.array([[0,0,0,0],])

def updateData():
    global axes_list
    pygame.init()
    pygame.joystick.init()
    logitech_wheel = pygame.joystick.Joystick(0)
    logitech_wheel.init()
    axes = logitech_wheel.get_numaxes()

    for event in pygame.event.get():  # User did something
        pass
    logitech_wheel = pygame.joystick.Joystick(0)
    logitech_wheel.init()
    axex_values = []
    for num in range(axes):
        axex_values.append(logitech_wheel.get_axis(num))
    axes_list = np.vstack((axes_list,np.array(axex_values)))
        # print('axes',axes_list,end='\r')
    updatePlot(axes_list)


timerdata = QtCore.QTimer()
timerdata.timeout.connect(updateData)
timerdata.start()

pygame.joystick.quit()
pygame.quit()

def updatePlot(axes_list):
    p1.clear()
    p1.plot(axes_list[-150:,0]*2., pen=(255, 255, 255))
    p1.plot(axes_list[-150:,1], pen=(0, 255, 0))
    p1.plot(axes_list[-150:,2], pen=(255, 0, 0))
    p1.plot(axes_list[-150:,3], pen=(0, 0, 255))


if __name__ == '__main__':
#    evaluate()
    import sys
    try:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    except:
        print('Plotting GUI doesn\'t exist \n')






































# import pygame
#
# # Define some colors
# BLACK = (0, 0, 0)
# WHITE = (255, 255, 255)
#
#
# # This is a simple class that will help us print to the screen
# # It has nothing to do with the joysticks, just outputting the
# # information.
# class TextPrint:
#     def __init__(self):
#         self.reset()
#         self.font = pygame.font.Font(None, 20)
#
#     def print(self, screen, textString):
#         textBitmap = self.font.render(textString, True, BLACK)
#         screen.blit(textBitmap, [self.x, self.y])
#         self.y += self.line_height
#
#     def reset(self):
#         self.x = 10
#         self.y = 10
#         self.line_height = 15
#
#     def indent(self):
#         self.x += 10
#
#     def unindent(self):
#         self.x -= 10
#
#
# pygame.init()
#
# # Set the width and height of the screen [width,height]
# size = [500, 700]
# screen = pygame.display.set_mode(size)
#
# pygame.display.set_caption("My Game")
#
# # Loop until the user clicks the close button.
# done = False
#
# # Used to manage how fast the screen updates
# clock = pygame.time.Clock()
#
# # Initialize the joysticks
# pygame.joystick.init()
#
# # Get ready to print
# textPrint = TextPrint()
#
# # -------- Main Program Loop -----------
# while done == False:
#     # EVENT PROCESSING STEP
#     for event in pygame.event.get():  # User did something
#         pass
#         # if event.type == pygame.QUIT:  # If user clicked close
#         #     done = True  # Flag that we are done so we exit this loop
#
#         # # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
#         # if event.type == pygame.JOYBUTTONDOWN:
#         #     print("Joystick button pressed.")
#         # if event.type == pygame.JOYBUTTONUP:
#         #     print("Joystick button released.")
#
#     # DRAWING STEP
#     # First, clear the screen to white. Don't put other drawing commands
#     # above this, or they will be erased with this command.
#     # screen.fill(WHITE)
#     # textPrint.reset()
#
#     # Get count of joysticks
#     joystick_count = pygame.joystick.get_count()
#
#     # textPrint.print(screen, "Number of joysticks: {}".format(joystick_count))
#     # textPrint.indent()
#
#     # For each joystick:
#     for i in range(joystick_count):
#         joystick = pygame.joystick.Joystick(i)
#         joystick.init()
#
#         # textPrint.print(screen, "Joystick {}".format(i))
#         # textPrint.indent()
#         #
#         # # Get the name from the OS for the controller/joystick
#         # name = joystick.get_name()
#         # textPrint.print(screen, "Joystick name: {}".format(name))
#         #
#         # # Usually axis run in pairs, up/down for one, and left/right for
#         # # the other.
#         axes = joystick.get_numaxes()
#         # textPrint.print(screen, "Number of axes: {}".format(axes))
#         # textPrint.indent()
#
#         for i in range(axes):
#             axis = joystick.get_axis(i)
#             print('axis',i, axis)
#         #     textPrint.print(screen, "Axis {} value: {:>6.3f}".format(i, axis))
#         # textPrint.unindent()
#
#         buttons = joystick.get_numbuttons()
#         # textPrint.print(screen, "Number of buttons: {}".format(buttons))
#         # textPrint.indent()
#
#         for i in range(buttons):
#             button = joystick.get_button(i)
#             # textPrint.print(screen, "Button {:>2} value: {}".format(i, button))
#         # textPrint.unindent()
#
#         # Hat switch. All or nothing for direction, not like joysticks.
#         # Value comes back in an array.
#         hats = joystick.get_numhats()
#         # textPrint.print(screen, "Number of hats: {}".format(hats))
#         # textPrint.indent()
#
#         for i in range(hats):
#             hat = joystick.get_hat(i)
#             textPrint.print(screen, "Hat {} value: {}".format(i, str(hat)))
#         # textPrint.unindent()
#
#         # textPrint.unindent()
#
#     # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT
#
#     # Go ahead and update the screen with what we've drawn.
#     # pygame.display.flip()
#
#     # Limit to 20 frames per second
#     # clock.tick(20)
#
# # Close the window and quit.
# # If you forget this line, the program will 'hang'
# # on exit if running from IDLE.
# pygame.quit()