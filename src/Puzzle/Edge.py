import numpy as np

from Puzzle.Enums import TypePiece, Directions


class Edge:
    def __init__(self, shape, color, type=TypePiece.HOLE, connected=False, direction=Directions.N):
        self.shape = shape
        self.shape_backup = shape
        self.color = color
        self.type = type
        self.connected = connected
        self.direction = direction

    def is_border(self, threshold):
        def dist_to_line(p1, p2, p3):
            return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

        total_dist = 0
        for p in self.shape:
            total_dist += dist_to_line(self.shape[0], self.shape[-1], p)
        return total_dist < threshold

    def backup_shape(self):
        self.shape_backup = np.copy(self.shape)

    def restore_backup_shape(self):
        self.shape = self.shape_backup
