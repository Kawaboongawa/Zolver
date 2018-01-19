import numpy as np

from Puzzle.Enums import directions, Directions, TypeEdge, TypePiece, rotate_direction

class PuzzlePiece():
    def __init__(self, edges, img_piece):
        self.position = (0, 0)
        self.edges_ = edges
        self.img_piece_ = img_piece  # List of Pixels
        self.nBorders_ = self.number_of_border()
        self.type = TypePiece(self.nBorders_)

    def number_of_border(self):
        return len(list(filter(lambda x: x.type == TypeEdge.BORDER, self.edges_)))

    def rotate_edges(self, r):
        for e in self.edges_:
            e.direction = rotate_direction(e.direction, r)

    def edge_in_direction(self, dir):
        for e in self.edges_:
            if e.direction == dir:
                return e

    def is_border_aligned(self, p2):
        for e in self.edges_:
            if e.type == TypeEdge.BORDER and p2.edge_in_direction(e.direction).type == TypeEdge.BORDER:
                return True
        return False