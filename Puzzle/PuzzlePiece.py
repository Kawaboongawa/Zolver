import numpy as np

from .Enums import TypeEdge, TypePiece, rotate_direction
from .utils import rotate


class PuzzlePiece:
    """
    Wrapper used to store informations about pieces of the puzzle.
    Contains the position of the piece in the puzzle graph, a list of edges,
    the list of pixels composing the piece, the number of borders and the type
    of the piece.
    """

    def __init__(self, edges, pixels):
        self.position = (0, 0)
        self.edges_ = edges
        self.pixels = pixels
        self.nBorders_ = self.number_of_border()
        self.type = TypePiece(self.nBorders_)
        self.is_border = self.number_of_border() > 0

    def get_bbox(self):
        x = list(map(lambda p: p[0], self.pixels))
        y = list(map(lambda p: p[1], self.pixels))
        return int(min(x)), int(min(y)), int(max(x)), int(max(y))

    def rotate_bbox(self, angle, around):
        # Rotate corners only to optimize
        minX, minY, maxX, maxY = self.get_bbox()
        rotated = [
            rotate((x, y), angle, around) for x in [minX, maxX] for y in [minY, maxY]
        ]
        rotatedX = [p[0] for p in rotated]
        rotatedY = [p[1] for p in rotated]
        return (
            int(min(rotatedX)),
            int(min(rotatedY)),
            int(max(rotatedX)),
            int(max(rotatedY)),
        )

    def get_center(self):
        minX, minY, maxX, maxY = self.get_bbox()
        return ((minX + maxX) // 2, (minY + maxY) // 2)

    def translate(self, dx, dy):
        self.pixels = {(x + dx, y + dy): c for (x, y), c in self.pixels.items()}

    def rotate(self, angle, around):
        self.pixels = {
            rotate((x, y), angle, around, to_int=True): c
            for (x, y), c in self.pixels.items()
        }

    def get_image(self):
        minX, minY, maxX, maxY = self.get_bbox()
        img_p = np.full((maxX - minX + 1, maxY - minY + 1, 3), -1)
        for (x, y), c in self.pixels.items():
            img_p[x - minX, y - minY] = c
        return img_p

    def number_of_border(self):
        """Fast computations of the number of borders"""

        return len(list(filter(lambda x: x.type == TypeEdge.BORDER, self.edges_)))

    def rotate_edges(self, r):
        """Rotate the edges"""

        for e in self.edges_:
            e.direction = rotate_direction(e.direction, r)

    def edge_in_direction(self, dir):
        """Return the edge in the `dir` direction"""

        for e in self.edges_:
            if e.direction == dir:
                return e

    def is_border_aligned(self, p2):
        """Find if a border of the piece is aligned with a border of `p2`"""

        for e in self.edges_:
            if (
                e.type == TypeEdge.BORDER
                and p2.edge_in_direction(e.direction).type == TypeEdge.BORDER
            ):
                return True
        return False
