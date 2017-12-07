import numpy as np

from Img.FourierDescriptor import FourierDescriptor
from Puzzle.Enums import directions, Directions, TypePiece


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x) * 180 / np.pi
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y







def normalize_edge(edge, n):
    point_dist = float(len(edge)) / float(n)
    index = float(0)
    dst = []
    for i in range(0, n):
        # TODO: fix triple array
        dst.append([edge[int(index)][0], edge[int(index)][1]])
        index += point_dist
    return dst

def normalize_list(l, n):
    point_dist = float(len(l)) / float(n)
    index = float(0)
    dst = []
    for i in range(0, n):
        # TODO: fix triple array
        dst.append(l[int(index)])
        index += point_dist
    return dst

class PuzzlePiece():
    def __init__(self, edges, img_piece):
        self.position = (0, 0)
        # Keep orientations in an array (Correct only for first piece then the
        # values will be ovewritten)
        self.relative_angles_ = []
        self.edges_ = edges
        self.img_piece_ = img_piece  # List of Pixels
        self.nBorders_ = self.number_of_border()


    def normalize_edges(self, edge, n):
        point_dist = float(len(edge)) / float(n)
        index = float(0)
        dst = []
        for i in range(0, n):
            # TODO: fix triple array
            dst.append((edge[int(index)][0][0], edge[int(index)][0][1]))
            index += point_dist
        return dst

    def number_of_border(self):
        return len(list(filter(lambda x: x.type == TypePiece.BORDER, self.edges_)))