import numpy as np
import matplotlib.pyplot as plt


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x) * 180 / np.pi
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


class PuzzlePiece():
    def __init__(self, edges, corners):
        self.corners = corners
        self.edges_ = edges

    def normalize_edges(self, n, edge):
        point_dist = float(len(edge)) / float(n)
        index = float(0)
        dst = [tuple(0, 0)] * n
        for i in range(0, n):
            dst[i] = edge[int(index)]
            index += point_dist
        return dst

    edges_ = None
    n_corners = 0
    pol_edges = None
