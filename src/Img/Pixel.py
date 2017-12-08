import math
import numpy as np


class Pixel:
    def __init__(self, pos, color):
        self.pos = pos
        self.color = color

    def apply(self, img, dx=0, dy=0):
        x, y = self.pos
        x, y = x + dx, y + dy
        if x >= 0 and y >= 0 and x < img.shape[0] and y < img.shape[1]:
            img[(x, y)] = self.color

    def translate(self, dx, dy):
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def rotate(self, origin, angle):
        ox, oy = origin
        py, px = self.pos

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        if qx != qx or qy != qy:
            print("NAN DETECTED: {} {} {} {} {}".format(ox, oy, px, py, qx, qy, angle))

        self.pos = (int(qy), int(qx))

def flatten_colors(pixels):
    colors = np.array(pixels)
    return np.median(colors, axis=0)

def compare_colors(c1, c2):
    diff = np.abs(c1 - c2)
    return np.sum(diff)
