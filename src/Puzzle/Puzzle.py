from Puzzle.PuzzlePiece import *

from Puzzle.Extractor import Extractor
from Puzzle.Mover import *
from cv2 import cv2

class Puzzle():
    def __init__(self, path, pixmapWidget):
        self.extract = Extractor(path, pixmapWidget)
        self.pieces_ = self.extract.extract()

        self.stick_best(self.pieces_[0], 3)
        self.stick_best(self.pieces_[0], 2)

        self.export_pieces("/tmp/test_stick.png")

    def stick_best(self, cur_piece, edge_cur_piece):
        tests = []
        for index_piece, piece in enumerate(self.pieces_):
            if piece != cur_piece:
                for index_edge, edge in enumerate(piece.edges_):
                    tests.append((index_piece, index_edge, piece.fourier_descriptors_[index_edge].match_descriptors(cur_piece.fourier_descriptors_[edge_cur_piece])))

        l = sorted(tests, key=lambda x: x[2])
        stick_pieces(cur_piece, edge_cur_piece, self.pieces_[l[0][0]], l[0][1])

    def export_pieces(self, path):
        tests_img = np.zeros_like(self.extract.img)

        for piece in self.pieces_:
            for i in range(4):
                for p in piece.edges_[i]:
                    if not piece.border[i] and p[0][0] < self.extract.img.shape[1] and p[0][1] < self.extract.img.shape[0]:
                        tests_img[p[0][1], p[0][0]] = 255

        # cv2.circle(tests_img, tuple((int(puzzle_pieces[1].edges_[0][0][0]), int(centerY))), 10, 255, -1)
        cv2.imwrite(path, tests_img)
