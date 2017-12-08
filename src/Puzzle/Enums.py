from enum import Enum

# Directions used to keep track of orientation of edges
class Directions(Enum):
    N = (0, 1)
    S = (0, -1)
    E = (1, 0)
    W = (-1, 0)

directions = [Directions.N, Directions.E, Directions.S, Directions.W]

def rotate_direction(dir, step):
    i = directions.index(dir)
    return directions[(i + step) % 4]

def step_direction(dir1, dir2):
    return (directions.index(dir1) - directions.index(dir2) + 4) % 4

def get_opposite_direction(dir1):
    for dir2 in directions:
        if dir1.value[0] == -dir2.value[0] and dir1.value[1] == -dir2.value[1]:
            return dir2

class TypeEdge(Enum):
    HOLE = 0
    HEAD = 1
    BORDER = 2
    UNDEFINED = 3
