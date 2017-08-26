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
    def __init__(self, edges):
        self.edges_ = edges
        self.pol_edges = []
        print(edges)
        for e in self.edges_:
            # Checker x, y et ou y, x !
            # NE MARCHE PAAAAAAAAAAAAAAS
            self.pol_edges.append(cart2pol(e.item(0), e.item(1)))
        # print(self.pol_edges)
        # plt.imshow(self.pol_edges)
        # plt.show()

    edges_ = None
    n_corners = 0
    pol_edges = None
