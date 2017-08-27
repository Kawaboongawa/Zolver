from Puzzle.PuzzlePiece import *

from src.Puzzle.Extractor import Extractor


class Puzzle():
    def __init__(self, path, pixmapWidget):
        self.extract = Extractor(path, pixmapWidget)
        self_pieces = self.extract.extract()


    n_pieces_ = 0
