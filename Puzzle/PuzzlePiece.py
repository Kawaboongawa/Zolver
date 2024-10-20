from .Enums import TypeEdge, TypePiece, rotate_direction


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

    def get_bbox(self):
        x = list(map(lambda p: p[0], self.pixels))
        y = list(map(lambda p: p[1], self.pixels))
        return min(x), min(y), max(x), max(y)

    def translate(self, dx, dy):
        self.pixels = {(x + dx, y + dy): c for (x, y), c in self.pixels.items()}

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
