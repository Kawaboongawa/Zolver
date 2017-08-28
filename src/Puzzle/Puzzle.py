from Puzzle.PuzzlePiece import *

from Puzzle.Extractor import Extractor
from Puzzle.Mover import *
from cv2 import cv2

def diff_match_edges(e1, e2):
    diff = 0
    for i, p in enumerate(e1):
        if i < len(e2):
            diff += np.linalg.norm(p[0] - e2[len(e2) - i - 1][0])
        else:
            break
    return diff

class Puzzle():
    def __init__(self, path, pixmapWidget):
        self.extract = Extractor(path, pixmapWidget)
        self.pieces_ = self.extract.extract()

        # stick_pieces(self.pieces_[1], 3, self.pieces_[8], 1)
        self.stick_best(self.pieces_[1], 2)
        self.stick_best(self.pieces_[1], 3)
        self.stick_best(self.pieces_[4], 3)
        self.stick_best(self.pieces_[4], 2)
        self.stick_best(self.pieces_[8], 3)
        self.stick_best(self.pieces_[3], 3)
        self.stick_best(self.pieces_[6], 3)
        self.stick_best(self.pieces_[0], 0)

        self.export_pieces("/tmp/test_stick.png")

    def stick_best(self, cur_piece, edge_cur_piece):
        if cur_piece.border[edge_cur_piece]:
            return

        tests = []
        for index_piece, piece in enumerate(self.pieces_):
            if piece != cur_piece:
                for index_edge, edge in enumerate(piece.edges_):
                    tests.append((index_piece, index_edge, piece.fourier_descriptors_[index_edge].match_descriptors(cur_piece.fourier_descriptors_[edge_cur_piece])))

        l = sorted(tests, key=lambda x: x[2])
        diff = []
        for i in range(len(l)):
            if not self.pieces_[l[i][0]].border[l[i][1]]:
                tmp = np.array(self.pieces_[l[i][0]].edges_)
                for j in range(4):
                    tmp[j] = np.array(self.pieces_[l[i][0]].edges_[j])

                stick_pieces(cur_piece, edge_cur_piece, self.pieces_[l[i][0]], l[i][1])
                diff.append(diff_match_edges(self.pieces_[l[i][0]].edges_[l[i][1]], cur_piece.edges_[edge_cur_piece]))
                self.pieces_[l[i][0]].edges_ = tmp
            else:
                diff.append(float('inf'))

        m = np.argmin(diff)
        print(l[m][0], l[m][1])
        stick_pieces(cur_piece, edge_cur_piece, self.pieces_[l[m][0]], l[m][1])

    def export_pieces(self, path):
        tests_img = np.zeros_like(self.extract.img)

        for piece in self.pieces_:
            for i in range(4):
                for p in piece.edges_[i]:
                    # if not piece.border[i] and p[0][0] < self.extract.img.shape[1] and p[0][1] < self.extract.img.shape[0]:
                    if p[0][0] < self.extract.img.shape[1] and p[0][1] < self.extract.img.shape[0]:
                        tests_img[p[0][1], p[0][0]] = 255

        # cv2.circle(tests_img, tuple((int(puzzle_pieces[1].edges_[0][0][0]), int(centerY))), 10, 255, -1)
        cv2.imwrite(path, tests_img)
