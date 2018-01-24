import colorsys

from cv2 import cv2

import numpy as np
import math
from skimage import color
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt

def dist(p1, p2):
    """
        Compute euclidean distance

        :param p1: first coordinate tuple
        :param p2: second coordinate tuple
        :return: distance Float
    """

    return math.sqrt((p2[0] - p1[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def dist_edge(e1, e2):
    """
        Compute the size difference between two edges

        :param e1: Matrix of coordinates of points composing the first edge
        :param e2: Matrix of coordinates of points composing the second edge
        :return: Float
    """
    e1_begin = e1.shape[0]
    e1_end = e1.shape[-1]
    e2_begin = e2.shape[0]
    e2_end = e2.shape[-1]
    dist_e1 = dist(e1_begin, e1_end)
    dist_e2 = dist(e2_begin, e2_end)
    res = math.fabs(dist_e1 - dist_e2)
    val = (dist_e1 + dist_e2) / 2
    return res, val
    
def have_edges_similar_length(e1, e2, percent):
    """
        Return a boolean to determine if the difference between two edges is > 20%

        :param e1: Matrix of coordinates of points composing the first edge
        :param e2: Matrix of coordinates of points composing the second edge
        :return: Boolean
    """
    res, val = dist_edge(e1, e2)    
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
        pad_left, pad_right = pad_length, (pad_length if pad_length * 2 == (e1.shape[0] - e2.shape[0]) else pad_length + 1)

        # Pad the shortest with 0
        e2 = np.lib.pad(e2, ((pad_left, pad_right), (0, 0)), 'constant', constant_values=(0, 0))
    else:
        # No padding just cut longest to match shortest length
        e1 = e1[:e2.shape[0]]

    if reverse:
        e2 = np.flip(e2, 0)
    d = np.linalg.norm(e1 - e2, axis=1)
    return np.sum(d > thres) / e1.shape[0]

def euclideanDistance(e1_lab_colors, e2_lab_colors):
    sum = 0
    max = 50
    len1 = len(e1_lab_colors)
    len2 = len(e2_lab_colors)
    if len1 < len2:
        max = len1
    else:
        max = len2
    t1 = len1 / max
    t2 = len2 / max

    def dist_color(tuple1, tuple2):
        return np.sqrt((tuple1[0] - tuple2[0]) ** 2
                        + (tuple1[1] - tuple2[1]) ** 2
                        + (tuple1[2] - tuple2[2]) ** 2)

    for i in range(max):
        sum += dist_color(e1_lab_colors[int(t1 * i)], e2_lab_colors[int(t2 * i)])
    return sum

def real_edge_compute(e1, e2):
    """
        Return the distance between colors of two edges for real puzzle.

        :param e1: Edge object
        :param e2: Edge object
        :return: distance Float
    """
    
    rgbs1 = []
    rgbs2 = []
    if not have_edges_similar_length(e1, e2, 0.20):
        return float('inf')

    e1_lab_colors = []
    for col in e1.color:
        rgb = colorsys.hls_to_rgb(col[0], col[1], col[2])
        rgb = [x * 255.0 for x in rgb]
        rgbs1.append(rgb)
        e1_lab_colors.append(color.rgb2lab([[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]])[0][0])
        # Drop luminance
        e1_lab_colors[-1] = [0, e1_lab_colors[-1][1], e1_lab_colors[-1][2]]

    e2_lab_colors = []
    for col in e2.color:
        rgb = colorsys.hls_to_rgb(col[0], col[1], col[2])
        rgb = [x * 255.0 for x in rgb]
        rgbs2.append(rgb)
        e2_lab_colors.append(color.rgb2lab([[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]])[0][0])
        # Drop Luminance
        e2_lab_colors[-1] = [0, e2_lab_colors[-1][1], e2_lab_colors[-1][2]]

    return min(euclideanDistance(e1_lab_colors, e2_lab_colors), euclideanDistance(e1_lab_colors, e2_lab_colors[::-1]))


def generated_edge_compute(e1, e2):
    """
        Return the distance between colors of two edges for generated puzzle.

        :param e1: Edge object
        :param e2: Edge object
        :return: distance Float
    """
    #edge size
    shapevalue, distvalue = dist_edge(e1, e2) 

    #edges diff

    edge_shape_score = diff_match_edges2(np.array(e1.shape), np.array(e2.shape))
     # Sigmoid
    L = 10
    K = -1.05
    #edge_color_score = 1 / (1 + math.exp(-L * (edge_color_score - 0.5)))
    edge_shape_score = (K * edge_shape_score) / (K - edge_shape_score + 1)

    #colors
    rgbs1 = []
    rgbs2 = []

    e1_lab_colors = []
    for col in e1.color:
        rgb = colorsys.hls_to_rgb(col[0], col[1], col[2])
        rgb = [x * 255.0 for x in rgb]
        rgbs1.append(rgb)
        e1_lab_colors.append(color.rgb2lab([[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]])[0][0])
        # Drop luminance
        e1_lab_colors[-1] = [0, e1_lab_colors[-1][1], e1_lab_colors[-1][2]]

    e2_lab_colors = []
    for col in e2.color:
        rgb = colorsys.hls_to_rgb(col[0], col[1], col[2])
        rgb = [x * 255.0 for x in rgb]
        rgbs2.append(rgb)
        e2_lab_colors.append(color.rgb2lab([[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]])[0][0])
        # Drop Luminance
        e2_lab_colors[-1] = [0, e2_lab_colors[-1][1], e2_lab_colors[-1][2]]

    val = min(euclideanDistance(e1_lab_colors, e2_lab_colors), euclideanDistance(e1_lab_colors, e2_lab_colors[::-1]))
    return val * (1.0 + math.sqrt(shapevalue) * 0.3) * (1.0 + edge_shape_score * 0.001)
