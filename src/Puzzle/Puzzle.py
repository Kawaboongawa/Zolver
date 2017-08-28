from Puzzle.PuzzlePiece import *

from src.Puzzle.Extractor import Extractor


class Puzzle():
    def __init__(self, path, pixmapWidget):
        self.extract = Extractor(path, pixmapWidget)
        self.pieces_ = self.extract.extract()
        print(self.pieces_[0].fourier_descriptors_[0].match_descriptors(self.pieces_[1].fourier_descriptors_[1]))
