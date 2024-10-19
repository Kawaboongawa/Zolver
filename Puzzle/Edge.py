import numpy as np

from .Enums import TypeEdge, Directions


class Edge:
    """
    Wrapper for edges.
    Contains shape, colors, type and positions informations in the puzzle of an edge.
    """

    def __init__(
        self,
        shape,
        color,
        edge_type=TypeEdge.HOLE,
        connected=False,
        direction=Directions.N,
    ):
        self.shape = shape
        self.shape_backup = shape
        self.color = color
        self.type = edge_type
        self.connected = connected
        self.direction = direction

    def backup_shape(self):
        """Copy the shape for backup"""
        self.shape_backup = np.copy(self.shape)

    def restore_backup_shape(self):
        """Restore the shape previously backedup"""
        self.shape = self.shape_backup

    def is_compatible(self, e2):
        """Helper to determine if two edges are compatible"""
        return (
            (self.type == TypeEdge.HOLE and e2.type == TypeEdge.HEAD)
            or (self.type == TypeEdge.HEAD and e2.type == TypeEdge.HOLE)
            or self.type == TypeEdge.UNDEFINED
            or e2.type == TypeEdge.UNDEFINED
        )
