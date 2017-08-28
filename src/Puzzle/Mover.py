from Puzzle.PuzzlePiece import *
from Img.filters import angle_between
import math

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def stick_pieces(bloc, bloc_index_edge, piece, piece_index_edge):
    vec_bloc = bloc.edges_[bloc_index_edge][0][0] - bloc.edges_[bloc_index_edge][-1][0]
    vec_piece = piece.edges_[piece_index_edge][0][0] - piece.edges_[piece_index_edge][-1][0]
    translation = bloc.edges_[bloc_index_edge][0][0] - piece.edges_[piece_index_edge][-1][0]
    angle = angle_between((vec_bloc[0], vec_bloc[1], 0), (-vec_piece[0], -vec_piece[1], 0))
    print(math.degrees(angle), bloc.edges_[bloc_index_edge][0][0], bloc.edges_[bloc_index_edge][-1][0])

    print("test1", math.degrees(angle_between((0, -1, 0), (1, 1, 0))))
    print("test2", math.degrees(angle_between((0, -1, 0), (1, 1, 0))))

    # First move the first corner of piece to the corner of bloc edge
    for index, edge in enumerate(piece.edges_):
        piece.edges_[index] += translation

    # Then rotate piece of `angle` degrees centered on the corner
    for index_edge, edge in enumerate(piece.edges_):
        for index_point, p in enumerate(edge):
            piece.edges_[index_edge][index_point][0] = rotate(bloc.edges_[bloc_index_edge][0][0], p[0], angle)
