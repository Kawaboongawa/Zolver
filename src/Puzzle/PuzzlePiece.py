import numpy as np

from Img.FourierDescriptor import FourierDescriptor


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
        self.edges_ = edges
        self.border = []
        self.fourier_descriptors_ = []
        for e in edges:
            norm_e = self.normalize_edges(e, 128)
            self.fourier_descriptors_.append(FourierDescriptor(norm_e, 128))
            self.border.append(is_border(e, 10))

    def normalize_edges(self, edge, n):
        point_dist = float(len(edge)) / float(n)
        index = float(0)
        dst = []
        for i in range(0, n):
            #TODO: fix triple array
            dst.append((edge[int(index)][0][0], edge[int(index)][0][1]))
            index += point_dist
        return dst
