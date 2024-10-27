import math

from numba import njit


@njit
def rotate(point, angle, around, to_int=False):
    """
    Rotate the pixel around `origin` by `angle` degrees

    :param around: Coordinates of the point used to rotate around
    :param point: Coordinates of the point to rotate
    :param angle: number of degrees
    :return: Coordinates after rotation
    """

    ox, oy = around
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    if to_int:
        qx, qy = int(qx), int(qy)
    return qx, qy


@njit
def angle_between(v1, v2):
    """
    Return the angles between two tuples

    :param v1: first tuple of coordinates
    :param v2: second tuple of coordinates
    :return: distance Float
    """

    return math.atan2(-v1[1], v1[0]) - math.atan2(-v2[1], v2[0])
