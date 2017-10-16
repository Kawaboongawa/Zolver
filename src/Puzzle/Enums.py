from enum import Enum

# Directions used to keep track of orientation of edges
class Directions(Enum):
    N = (0, 1)
    S = (0, -1)
    E = (1, 0)
    W = (-1, 0)
