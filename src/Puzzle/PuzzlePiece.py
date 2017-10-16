import numpy as np

from Img.FourierDescriptor import FourierDescriptor
from Puzzle.Enums import Directions

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x) * 180 / np.pi
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def dist_to_line(p1, p2, p3):
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

def is_border(edge, threshold):
    total_dist = 0
    for p in edge:
        total_dist += dist_to_line(edge[0][0], edge[-1][0], p[0])
    return total_dist < threshold

class PuzzlePiece():
    def __init__(self, edges):
        self.position = (0, 0)
        # Keep orientations in an array (Correct only for first piece then the
        # values will be ovewritten)
        self.orientation = [Directions.N, Directions.E, Directions.S, Directions.W]
        self.edges_ = edges

        # Keep informations if the edge is a connected
        self.connected_ = []

        self.fourier_descriptors_ = []
        for e in edges:
            norm_e = self.normalize_edges(e, 128)
            self.fourier_descriptors_.append(FourierDescriptor(norm_e, 128))
            self.connected_.append(is_border(e, 1000))

    def normalize_edges(self, edge, n):
        point_dist = float(len(edge)) / float(n)
        index = float(0)
        dst = []
        for i in range(0, n):
            #TODO: fix triple array
            dst.append((edge[int(index)][0][0], edge[int(index)][0][1]))
            index += point_dist
        return dst
