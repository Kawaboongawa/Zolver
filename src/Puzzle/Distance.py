import numpy as np
import math
from skimage import color
from colorsys import hls_to_rgb

def normalize_vect_len(e1, e2):
    longest = e1 if len(e1) > len(e2) else e2
    shortest = e2 if len(e1) > len(e2) else e1
    # indexes = np.array(range(len(longest)))
    # np.random.shuffle(indexes)
    # indexes = indexes[:len(shortest)]
    # longest = longest[sorted(indexes)]
    return shortest, longest


def diff_match_edges2(e1, e2, reverse=True):
    shortest, longest = normalize_vect_len(e1, e2)
    diff = 0
    for i, p in enumerate(shortest):
        ratio = i / len(shortest)
        j = int(len(longest) * ratio)
        x1, y1 = longest[j]
        x2, y2 = shortest[len(shortest) - i - 1] if reverse else shortest[i]
        diff += (x2 - x1) ** 2 + (y2 - y1) ** 2
    return diff / len(shortest)


def diff_match_edges(e1, e2, reverse=True):
    shortest, longest = normalize_vect_len(e1, e2)
    diff = 0
    for i, p in enumerate(shortest):
        ratio = i / len(shortest)
        j = int(len(longest) * ratio)
        x1 = longest[j]
        x2 = shortest[len(shortest) - i - 1] if reverse else shortest[i]
        diff += (x2 - x1) ** 2
    return diff / len(shortest)


# Match edges by performing a simple norm on each points
def old_diff_match_edges(e1, e2, reverse=True, thres=5, pad=False):
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

def diff_colors(e1_lab_colors, e2_lab_colors, reverse=True, JNC=2.3, pad=False):
    #print(e1_lab_colors.shape[0], e2_lab_colors.shape[0])
    # Swap values to have longer first
    if e2_lab_colors.shape[0] > e1_lab_colors.shape[0]:
        e1_lab_colors, e2_lab_colors = e2_lab_colors, e1_lab_colors
    
    if pad:
        pad_length = (e1_lab_colors.shape[0] - e2_lab_colors.shape[0]) // 2
        pad_left, pad_right = pad_length, (pad_length if pad_length * 2 == (e1_lab_colors.shape[0] - e2_lab_colors.shape[0]) else pad_length + 1)

        # Pad the shortest with 0
        e2_lab_colors = np.lib.pad(e2_lab_colors, ((pad_left, pad_right), (0, 0)), 'constant', constant_values=(0, 0))
    else:
        # No padding just cut longest to match shortest length
        e1_lab_colors = e1_lab_colors[:e2_lab_colors.shape[0]]

    if reverse:
        e2_lab_colors = np.flip(e2_lab_colors, 0)
    
    d = np.linalg.norm(e1_lab_colors - e2_lab_colors, axis=1)
    return np.sum(d > JNC) / e1_lab_colors.shape[0] # > 2.3 -> Just noticeable difference

def diff_full_compute(e1, e2):
    e1_lab_colors = []
    for col in e1.color:
        rgb = hls_to_rgb(col[0], col[1], col[2])
        e1_lab_colors.append(color.rgb2lab([[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]])[0][0])
        # Drop luminance
        e1_lab_colors[-1] = [0, e1_lab_colors[-1][1], e1_lab_colors[-1][2]]
    
    e2_lab_colors = []
    for col in e2.color:
        rgb = hls_to_rgb(col[0], col[1], col[2])
        e2_lab_colors.append(color.rgb2lab([[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]])[0][0])
        # Drop Luminance
        e2_lab_colors[-1] = [0, e2_lab_colors[-1][1], e2_lab_colors[-1][2]]

    edge_shape_score = old_diff_match_edges(np.array(e1.shape), np.array(e2.shape))
    edge_color_score = diff_colors(np.array(e1_lab_colors), np.array(e2_lab_colors))

    # Sigmoid
    L = 10
    K = -1.05
    #edge_color_score = 1 / (1 + math.exp(-L * (edge_color_score - 0.5)))
    edge_shape_score = (K * edge_shape_score) / (K - edge_shape_score + 1)

    # print(e1.type, e2.type, edge_color_score, edge_shape_score, (edge_color_score + edge_shape_score) / 2)
    # return edge_color_score
    #return edge_shape_score
    return (edge_color_score + edge_shape_score) / 2


    a, b, c, d = 5, 1, 5, 1  # TCMS
    a, b, c, d = 0, 1, 1, 1  # TCMS
    return a * diff_match_edges2(e1.shape, e2.shape) \
           + b * diff_match_edges(e1.color[:, 0], e2.color[:, 0]) \
           + c * diff_match_edges(e1.color[:, 1], e2.color[:, 1]) \
           + d * diff_match_edges(e1.color[:, 2], e2.color[:, 2])