from src.PuzzlePiece import *

class Puzzle():
    def __init__(self, contours):
        self.n_pieces_ = len(contours)
        for e in contours:
            self.pieces_.append(PuzzlePiece(e))
            print(len(e))
        print(self.n_pieces_)

    pieces_ = []
    n_pieces_ = 0
