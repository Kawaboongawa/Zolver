import colorsys
import math

import numpy as np
from numba import njit


@njit
def rgb2lab(r, g, b, drop_l=False):
    """Fast Implementation of rgb2lab function (skimage too slow)"""

    r = r / 255
    r = (((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92) * 100

    g = g / 255
    g = (((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92) * 100

    b = b / 255
    b = (((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92) * 100

    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    X = round(X, 4) / 95.047
    X = X**0.3333333333333333 if X > 0.008856 else (7.787 * X) + (16.0 / 116.0)

    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Y = round(Y, 4) / 100.0
    Y = Y**0.3333333333333333 if Y > 0.008856 else (7.787 * Y) + (16.0 / 116.0)

    Z = r * 0.0193 + g * 0.1192 + b * 0.9505
    Z = round(Z, 4) / 108.883
    Z = Z**0.3333333333333333 if Z > 0.008856 else (7.787 * Z) + (16.0 / 116.0)

    L = (116.0 * Y) - 16
    a = 500.0 * (X - Y)
    b = 200.0 * (Y - Z)

    return (round(L, 4) if not drop_l else 0.0, round(a, 4), round(b, 4))


@njit
def dist(p1, p2):
    """
    Compute euclidean distance

    :param p1: first coordinate tuple
    :param p2: second coordinate tuple
    :return: distance Float
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@njit
def dist_edge(e1_begin, e1_end, e2_begin, e2_end):
    """
    Compute the size difference between two edges

    :param e1: Matrix of coordinates of points composing the first edge
    :param e2: Matrix of coordinates of points composing the second edge
    :return: Float
    """
    dist_e1 = dist(e1_begin, e1_end)
    dist_e2 = dist(e2_begin, e2_end)
    res = math.fabs(dist_e1 - dist_e2)
    val = (dist_e1 + dist_e2) / 2
    return res, val


@njit
def have_edges_similar_length(e1_begin, e1_end, e2_begin, e2_end, percent):
    """
    Return a boolean to determine if the difference between two edges is > 20%

    :param e1: Matrix of coordinates of points composing the first edge
    :param e2: Matrix of coordinates of points composing the second edge
    :return: Boolean
    """
    res, val = dist_edge(e1_begin, e1_end, e2_begin, e2_end)
    return res < (val * percent)


def normalize_vect_len(e1, e2):
    """
    Return the shortest and the longest edges.

    :param e1: Matrix of coordinates of points composing the first edge
    :param e2: Matrix of coordinates of points composing the second edge
    :return: Matrix of coordinates, Matrix of coordinates
    """

    longest = e1 if len(e1) > len(e2) else e2
    shortest = e2 if len(e1) > len(e2) else e1
    return shortest, longest


def diff_match_edges(e1, e2, reverse=True):
    """
    Return the distance between two edges.

    :param e1: Matrix of coordinates of points composing the first edge
    :param e2: Matrix of coordinates of points composing the second edge
    :param reverse: Optional parameter to reverse the second edge
    :return: distance Float
    """

    shortest, longest = normalize_vect_len(e1, e2)
    diff = 0
    for i, p in enumerate(shortest):
        ratio = i / len(shortest)
        j = int(len(longest) * ratio)
        x1 = longest[j]
        x2 = shortest[len(shortest) - i - 1] if reverse else shortest[i]
        diff += (x2 - x1) ** 2
    return diff / len(shortest)


def diff_match_edges2(e1, e2, reverse=True, thres=5, pad=False):
    """
    Return the distance between two edges by performing a simple norm on each points.

        :param e1: Matrix of coordinates of points composing the first edge
        :param e2: Matrix of coordinates of points composing the second edge
        :param reverse: Optional parameter to reverse the second edge
        :return: distance Float
    """
    if e2.shape[0] > e1.shape[0]:
        e1, e2 = e2, e1

    if pad:
        pad_length = (e1.shape[0] - e2.shape[0]) // 2
        pad_left, pad_right = (
            pad_length,
            (
                pad_length
                if pad_length * 2 == (e1.shape[0] - e2.shape[0])
                else pad_length + 1
            ),
        )

        # Pad the shortest with 0
        e2 = np.lib.pad(
            e2, ((pad_left, pad_right), (0, 0)), "constant", constant_values=(0, 0)
        )
    else:
        # No padding just cut longest to match shortest length
        e1 = e1[: e2.shape[0]]

    if reverse:
        e2 = np.flip(e2, 0)
    d = np.linalg.norm(e1 - e2, axis=1)
    return np.sum(d > thres) / e1.shape[0]


@njit
def dist_color(t1_0, t1_1, t1_2, t2_0, t2_1, t2_2):
    return np.sqrt((t1_0 - t2_0) ** 2 + (t1_1 - t2_1) ** 2 + (t1_2 - t2_2) ** 2)


def euclidean_distance(e1_lab_colors, e2_lab_colors):
    len1 = len(e1_lab_colors)
    len2 = len(e2_lab_colors)
    maximum = max(len1, len2)
    t1 = len1 / maximum
    t2 = len2 / maximum
    return sum(
        [
            dist_color(*e1_lab_colors[int(t1 * i)], *e2_lab_colors[int(t2 * i)])
            for i in range(maximum)
        ]
    )


def get_colors(edge):
    return [
        rgb2lab(*colorsys.hls_to_rgb(col[0], col[1], col[2]), drop_l=True)
        for col in edge.color
    ]


def real_edge_compute(e1, e2):
    """
    Return the distance between colors of two edges for real puzzle.

    :param e1: Edge object
    :param e2: Edge object
    :return: distance Float
    """

    if not have_edges_similar_length(
        e1.shape[0], e1.shape[-1], e2.shape[0], e2.shape[-1], 0.20
    ):
        return float("inf")

    e1_lab_colors = get_colors(e1)
    e2_lab_colors = get_colors(e2)
    return min(
        euclidean_distance(e1_lab_colors, e2_lab_colors),
        euclidean_distance(e1_lab_colors, e2_lab_colors[::-1]),
    )


def generated_edge_compute(e1, e2):
    """
    Return the distance between colors of two edges for generated puzzle.

    :param e1: Edge object
    :param e2: Edge object
    :return: distance Float
    """
    # edge size
    shapevalue, distvalue = dist_edge(
        e1.shape[0], e1.shape[-1], e2.shape[0], e2.shape[-1]
    )

    # edges diff
    edge_shape_score = diff_match_edges2(np.array(e1.shape), np.array(e2.shape))
    # Sigmoid
    K = -1.05
    edge_shape_score = (K * edge_shape_score) / (K - edge_shape_score + 1)

    # colors
    e1_lab_colors = get_colors(e1)
    e2_lab_colors = get_colors(e2)
    val = min(
        euclidean_distance(e1_lab_colors, e2_lab_colors),
        euclidean_distance(e1_lab_colors, e2_lab_colors[::-1]),
    )
    return val * (1.0 + math.sqrt(shapevalue) * 0.3) * (1.0 + edge_shape_score * 0.001)
