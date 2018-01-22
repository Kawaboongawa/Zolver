from Puzzle.PuzzlePiece import *
from Img.filters import angle_between
from Img.Pixel import *
import math
import numpy as np

def rotate(origin, point, angle):
    """
        Rotate the pixel around `origin` by `angle` degrees

        :param origin: Coordinates of points used to rotate around
        :param angle: number of degrees
        :return: Nothing
    """

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    if qx != qx or qy != qy:
        print("NAN DETECTED: {} {} {} {} {}".format(ox, oy, px, py, qx, qy, angle))

    return qx, qy


def stick_pieces(bloc_p, bloc_e, p, e, final_stick=False):
    """
        Stick an edge of a piece to the bloc of already resolved pieces

        :param bloc_p: bloc of pieces already solved
        :param bloc_e: bloc of edges already solved
        :param p: piece to add to the bloc
        :param e: edge to stick
        :return: Nothing
    """

    vec_bloc = np.subtract(bloc_e.shape[0], bloc_e.shape[-1])
    vec_piece = np.subtract(e.shape[0], e.shape[-1])

    translation = np.subtract(bloc_e.shape[0], e.shape[-1])
    angle = angle_between((vec_bloc[0], vec_bloc[1], 0), (-vec_piece[0], -vec_piece[1], 0))

    # First move the first corner of piece to the corner of bloc edge
    for edge in p.edges_:
        edge.shape += translation

    # Then rotate piece of `angle` degrees centered on the corner
    for edge in p.edges_:
        for i, point in enumerate(edge.shape):
            edge.shape[i] = rotate(bloc_e.shape[0], point, -angle)

    if final_stick:
        #prev bounding box
        minX, minY, maxX, maxY = float('inf'), float('inf'), -float('inf'), -float('inf')
        for i, pixel in enumerate(p.img_piece_):
            x, y = p.img_piece_[i].translate(translation[1], translation[0])
            minX, minY, maxX, maxY = min(minX, x), min(minY, y), max(maxX, x), max(maxY, y)
            # pixel.rotate(bloc_e.shape[0], -angle)

        
        #rotation center
        img_p = np.full((maxX - minX + 1, maxY - minY + 1, 3), -1)
        for pix in p.img_piece_:
            x, y = pix.pos
            x, y = x - minX, y - minY
            img_p[x, y] = pix.color

        #new bounding box
        minX2, minY2, maxX2, maxY2 = float('inf'), float('inf'), -float('inf'), -float('inf')
        for x in [minX, maxX]:
            for y in [minY, maxY]:
                x2, y2 = rotate((bloc_e.shape[0][1], bloc_e.shape[0][0]), (x,y), angle)
                x2, y2 = int(x2), int(y2)
                minX2, minY2, maxX2, maxY2 = min(minX2, x2), min(minY2, y2), max(maxX2, x2), max(maxY2, y2)

        pixels = []
        for px in range(minX2, maxX2 + 1):
            for py in range(minY2, maxY2 + 1):
                qx, qy = rotate((bloc_e.shape[0][1], bloc_e.shape[0][0]), (px,py), -angle)
                qx, qy = int(qx), int(qy)
                if minX <= qx <= maxX and minY <= qy <= maxY and img_p[qx - minX, qy - minY][0] != -1:
                    pixels.append(Pixel((px, py), img_p[qx - minX, qy - minY]))
                
        p.img_piece_ = pixels

