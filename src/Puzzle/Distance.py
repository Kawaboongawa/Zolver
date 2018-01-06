import colorsys

from cv2 import cv2

import numpy as np
import math
from skimage import color
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt

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

def show_image(img, ind=None, name='image', show=True):
    plt.axis("off")
    plt.imshow(img)
    if show:
        plt.show()

diffId = 0

def show_multiple_images(imgs, score, save=0):
    fig = plt.figure("Images")
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(len(imgs), i + 1, 1)
        ax.set_title(str(i) + ' : ' + str(score))
        show_image(img, show=False)
    if save == 1:
        global diffId
        fig.savefig("/tmp/diff" + str(diffId))
        if diffId < 100:
            diffId += 1
    else:
        plt.show()

def diff_full_compute(e1, e2):
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

    # edge_shape_score = old_diff_match_edges(np.array(e1.shape), np.array(e2.shape))
    # edge_color_score = diff_colors(np.array(e1_lab_colors), np.array(e2_lab_colors))
    # edge_color_score2 = diff_colors(np.array(e1_lab_colors), np.array(e2_lab_colors[::-1]))

    # Sigmoid
    # L = 10
    # K = -1.05
    # edge_color_score = 1 / (1 + math.exp(-L * (edge_color_score - 0.5)))
    # edge_shape_score = (K * edge_shape_score) / (K - edge_shape_score + 1)

    # print(e1.type, e2.type, edge_color_score, edge_shape_score, (edge_color_score + edge_shape_score) / 2)
    # return edge_color_score
    #return edge_shape_score
    # score = (edge_color_score + edge_shape_score) / 2
    # print('color:', edge_color_score, 'edge:', edge_shape_score)
    # print('score:', score)

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
    sum = euclideanDistance(e1_lab_colors, e2_lab_colors)
    sum2 = euclideanDistance(e1_lab_colors, e2_lab_colors[::-1])
    if sum2 < sum:
        sum = sum2
    # print('old', edge_color_score, 'new', sum)

    for i, _ in enumerate(rgbs1):
        rgbs1[i] = [rgbs1[i]]
    for i, _ in enumerate(rgbs2):
        rgbs2[i] = [rgbs2[i]]
    rgbs1 = np.array(rgbs1, dtype=np.uint8)
    rgbs2 = np.array(rgbs2, dtype=np.uint8)
    rgbs2 = rgbs2[::-1]
    # print(edge_color_score, sum)
    # if edge_color_score < 0.99:
    # show_multiple_images([rgbs1, rgbs2], edge_color_score, save=0)
    return sum
