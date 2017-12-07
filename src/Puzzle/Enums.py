from enum import Enum

# Directions used to keep track of orientation of edges
class Directions(Enum):
    N = (0, 1)
    S = (0, -1)
    E = (1, 0)
    W = (-1, 0)

directions = [Directions.N, Directions.E, Directions.S, Directions.W]

class TypePiece(Enum):
    HOLE = 0
    HEAD = 1
    BORDER = 2
    UNDEFINED = 3
