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


class PuzzlePiece():
    def __init__(self, edges):
        self.edges_ = edges
        self.fourier_descriptors_ = []
<<<<<<< HEAD
        for e in edges:
            norm_e = self.normalize_edges(e, 256)
            self.fourier_descriptors_.append(FourierDescriptor(norm_e, 256))
=======
        # for e in edges:
        #     norm_e = self.normalize_edges(e, 256)
        #     self.fourier_descriptors_.append(FourierDescriptor(norm_e, 256))
        # print(self.fourier_descriptors_)
>>>>>>> fix: include fix

    def normalize_edges(self, edge, n):
        point_dist = float(len(edge)) / float(n)
        index = float(0)
        dst = []
        for i in range(0, n):
            #TODO: fix triple array
            dst.append((edge[int(index)][0][0], edge[int(index)][0][1]))
            index += point_dist
        return dst
