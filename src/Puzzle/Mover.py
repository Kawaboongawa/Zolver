from src.Puzzle.PuzzlePiece import *
from src.Img.filters import angle_between

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def stick_pieces(bloc, bloc_index_edge, piece, piece_index_edge):
    vec_bloc = bloc.edges[bloc_index_edge][0] - bloc.edges[bloc_index_edge][-1]
    vec_piece = piece.edges[piece_index_edge][0] - piece.edges[piece_index_edge][-1]
    translation = bloc.edges[bloc_index_edge][0] - piece.edges[piece_index_edge][-1]
    angle = angle_between((vec_bloc[0], vec_bloc[1], 0), (vec_piece[0], vec_piece[1], 0))

    # First move the first corner of piece to the corner of bloc edge
    for index, edge in enumerate(piece.edges):
        piece.edges[index] += translation

    # Then rotate piece of `angle` degrees centered on the corner
    for index_edge, edge in enumerate(piece.edges):
        for index_point, p in enumerate(edge):
            piece.edges[index_edge][index_point] = rotate(bloc.edges[bloc_index_edge][0], p, angle)
