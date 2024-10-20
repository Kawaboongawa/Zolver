import math

import numpy as np
from numba import njit

from Img.filters import angle_between


@njit
def rotate(origin, point, angle):
    """
    Rotate the pixel around `origin` by `angle` degrees

    :param origin: Coordinates of points used to rotate around
    :param angle: number of degrees
    :return: Coordinates after rotation
    """

    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def stick_pieces(bloc_e, p, e, final_stick=False):
    """
    Stick an edge of a piece to the bloc of already resolved pieces

    :param bloc_e: bloc of edges already solved
    :param p: piece to add to the bloc
    :param e: edge to stick
    :return: Nothing
    """

    vec_bloc = np.subtract(bloc_e.shape[0], bloc_e.shape[-1])
    vec_piece = np.subtract(e.shape[0], e.shape[-1])

    translation = np.subtract(bloc_e.shape[0], e.shape[-1])
    angle = angle_between(
        (vec_bloc[0], vec_bloc[1], 0), (-vec_piece[0], -vec_piece[1], 0)
    )

    # First move the first corner of piece to the corner of bloc edge
    for edge in p.edges_:
        edge.shape += translation

    # Then rotate piece of `angle` degrees centered on the corner
    for edge in p.edges_:
        for i, point in enumerate(edge.shape):
            edge.shape[i] = rotate(bloc_e.shape[0], point, -angle)

    if final_stick:
        # prev bounding box
        p.translate(translation[1], translation[0])
        minX, minY, maxX, maxY = p.get_bbox()

        # rotation center
        img_p = np.full((maxX - minX + 1, maxY - minY + 1, 3), -1)
        for (x, y), c in p.pixels.items():
            img_p[x - minX, y - minY] = c

        # new bounding box
        b_e0, b_e1 = bloc_e.shape[0][0], bloc_e.shape[0][1]
        rotated = [
            rotate((b_e1, b_e0), (x, y), angle)
            for x in [minX, maxX]
            for y in [minY, maxY]
        ]
        rotatedX = [p[0] for p in rotated]
        rotatedY = [p[1] for p in rotated]
        minX2, minY2, maxX2, maxY2 = (
            int(min(rotatedX)),
            int(min(rotatedY)),
            int(max(rotatedX)),
            int(max(rotatedY)),
        )

        pixels = {}
        for px in range(minX2, maxX2 + 1):
            for py in range(minY2, maxY2 + 1):
                qx, qy = rotate((b_e1, b_e0), (px, py), -angle)
                qx, qy = int(qx), int(qy)
                if (
                    minX <= qx <= maxX
                    and minY <= qy <= maxY
                    and img_p[qx - minX, qy - minY][0] != -1
                ):
                    pixels[(px, py)] = img_p[qx - minX, qy - minY]
        p.pixels = pixels
