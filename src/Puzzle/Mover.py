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


def stick_pieces(bloc, bloc_index_edge, piece, piece_index_edge, final_stick=False):
    vec_bloc = np.subtract(bloc.edges_[bloc_index_edge][0][0], bloc.edges_[bloc_index_edge][0][-1])
    vec_piece =  np.subtract(piece.edges_[piece_index_edge][0][0], piece.edges_[piece_index_edge][0][-1])
    translation =  np.subtract(bloc.edges_[bloc_index_edge][0][0], piece.edges_[piece_index_edge][0][-1])
    angle = angle_between((vec_bloc[0], vec_bloc[1], 0), (-vec_piece[0], -vec_piece[1], 0))

    # First move the first corner of piece to the corner of bloc edge
    for index, edge in enumerate(piece.edges_):
        piece.edges_[index] += translation

    # Then rotate piece of `angle` degrees centered on the corner
    for index_edge, edge in enumerate(piece.edges_):
        for index_point, p in enumerate(edge):
            piece.edges_[index_edge][index_point][0] = rotate(bloc.edges_[bloc_index_edge][0][0], p[0], -angle)

    if final_stick:
        for p in piece.img_piece_:
            p.translate(translation[1], translation[0])
            p.rotate(bloc.edges_[bloc_index_edge][0][0], -angle)
