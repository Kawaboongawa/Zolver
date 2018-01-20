import math
import numpy as np


class Pixel:
    """
        Pixel wrapper.
        Contains the position and the color of the pixel
    """

    def __init__(self, pos, color):
        self.pos = pos
        self.color = color

    def apply(self, img, dx=0, dy=0):
        """
            Apply the pixel to an img wihth an optional translation

            :param img: image to apply the pixel
            :param dx: used for X axis translation
            :param dy: used for Y axis translation
            :return: Nothing
        """

        x, y = self.pos
        x, y = x + dx, y + dy
        if x >= 0 and y >= 0 and x < img.shape[0] and y < img.shape[1]:
            img[(x, y)] = self.color

    def translate(self, dx, dy):
        """
            Translate the pixel by (dx,dy)

            :param dx: used for X axis translation
            :param dy: used for Y axis translation
            :return: The updated position
        """

        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        return self.pos
    
    def rotate(self, origin, angle):
        """
            Rotate the pixel around `origin` by `angle` degrees

            :param origin: Coordinates of points used to rotate around
            :param angle: number of degrees
            :return: Nothing
        """

        ox, oy = origin
        py, px = self.pos

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        if qx != qx or qy != qy:
            print("NAN DETECTED: {} {} {} {} {}".format(ox, oy, px, py, qx, qy, angle))

        self.pos = (int(qy), int(qx))

def flatten_colors(pixels):
    """
        Return the median color

        :param pixels: list of colors
        :return: median Float
    """

    colors = np.array(pixels)
    return np.median(colors, axis=0)
