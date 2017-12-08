from Puzzle.PuzzlePiece import *
from Img.filters import angle_between
import math
import numpy as np

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    if qx != qx or qy != qy:
        print("NAN DETECTED: {} {} {} {} {}".format(ox, oy, px, py, qx, qy, angle))

    return qx, qy


def stick_pieces(bloc_p, bloc_e, p, e, final_stick=False):
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
        print(translation, angle)
        for pixel in p.img_piece_:
            pixel.translate(translation[1], translation[0])
            pixel.rotate(bloc_e.shape[0], -angle)

