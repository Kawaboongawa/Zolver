import numpy as np

from .utils import rotate, angle_between


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
            edge.shape[i] = rotate(point, -angle, bloc_e.shape[0])

    if final_stick:
        # Rotation origin
        b_e0, b_e1 = bloc_e.shape[0][0], bloc_e.shape[0][1]

        # Translate piece pixels to desired location
        p.translate(translation[1], translation[0])

        # Bounding boxes of origin/target space
        minX, minY, maxX, maxY = p.get_bbox()
        minX2, minY2, maxX2, maxY2 = p.rotate_bbox(angle, (b_e1, b_e0))

        # Recreate image from pixels
        img_p = p.get_image()

        # Retrieve new pixels by rotated target space into origin space
        pixels = {}
        for px in range(minX2, maxX2 + 1):
            for py in range(minY2, maxY2 + 1):
                # Rotate back to origin space
                qx, qy = rotate((px, py), -angle, (b_e1, b_e0))
                qx, qy = int(qx), int(qy)
                if (
                    minX <= qx <= maxX
                    and minY <= qy <= maxY
                    and img_p[qx - minX, qy - minY][0] != -1
                ):
                    pixels[(px, py)] = img_p[qx - minX, qy - minY]
        p.pixels = pixels
