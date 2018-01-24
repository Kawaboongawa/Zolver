import colorsys
from colorsys import rgb_to_hls

import cv2
import numpy as np
import math, pickle, os

from Img.Pixel import Pixel, flatten_colors
from Puzzle.Edge import Edge
from Puzzle.Enums import directions, TypeEdge
from Puzzle.PuzzlePiece import PuzzlePiece
import matplotlib.pyplot as plt
import matplotlib
import scipy, sklearn.preprocessing
import itertools
from Img.peak_detect import *

COUNT = 0

def get_relative_angles(cnt, export=False, sigma=5):
    """
        Get the relative angles of points of a contour 2 by 2

        :param cnt: contour to analyze
        :param export: export of the signature with pickle and figure
        :param sigma: coefficient used in gaussian filter (the higher the smoother)
        :type cnt: list of tuple of points
        :return: list of angles
    """

    global COUNT
    COUNT = COUNT + 1

    length = len(cnt)
    angles = []
    last = np.pi

    cnt_tmp = np.array(cnt)
    cnt = np.append(cnt, cnt_tmp, axis=0)
    cnt = np.append(cnt, cnt_tmp, axis=0)
    for i in range(0, len(cnt) - 1):
        dir = (cnt[i + 1][0] - cnt[i][0], cnt[i + 1][1] - cnt[i][1])
        angle = math.atan2(-dir[1], dir[0])
        while (angle < last - np.pi):
            angle += 2 * np.pi
        while (angle > last + np.pi):
            angle -= 2 * np.pi
        angles.append(angle)
        last = angle

    angles = np.diff(angles)

    k = [0.33,0.33,0.33,0.33,0.33]
    angles = scipy.ndimage.convolve(angles, k, mode='constant', cval=0.0)
    angles = scipy.ndimage.filters.gaussian_filter(angles, sigma)
    
    angles = np.roll(np.array(angles), -length)
    angles = angles[0:length]

    if export:
        pickle.dump(angles, open("/tmp/save" + str(COUNT) + ".p", "wb"))

        plt.plot(np.append(angles, angles))
        plt.savefig("/tmp/fig" + str(COUNT) + ".png")
        plt.clf()
        plt.cla()
        plt.close()

    return angles

def is_maximum_local(index, relative_angles, radius):
    """
        Determine if a point at index is a maximum local in radius range of relative_angles function

        :param index: index of the point to check in relative_angles list
        :param relative_angles: list of angles
        :param radius: radius used to check neighbors
        :return: Boolean
    """

    start = max(0, index - radius)
    end = min(relative_angles.shape[0] - 1, index + radius)
    for i in range(start, end + 1):
        if relative_angles[i] > relative_angles[index]:
            return False
    return True

def longest_peak(relative_angles):
    """
        Find the longest area < 0

        :param relative_angles: list of angles
        :return: coordinates of the area
    """

    length = relative_angles.shape[0]
    longest = (0, 0)
    j = 0
    for i in range(length):
        if relative_angles[i] >= 0:
            j = i
        if i - j > longest[1] - longest[0]:
            longest = (j, i)
    return longest

def distance_signature(relative_angles):
    """
        Distance of each points to the line formed by first and last points

        :param relative_angles: list of angles
        :return: List of floats
    """

    length = relative_angles.shape[0]
    
    l1 = np.array([0, relative_angles[0]])
    l2 = np.array([length - 1, relative_angles[-1]])
    
    signature = np.zeros((length, 1))

    for i in range(length):
        assert(np.linalg.norm(l2 - l1) != 0)
        signature[i] = np.linalg.norm(np.cross(l2 - l1, l1 - np.array([i, relative_angles[i]]))) / np.linalg.norm(l2 - l1)
    
    return signature

def flat_score(relative_angles):
    """
        Compute the flat score of relative_angles

        :param relative_angles: list of angles
        :return: List of floats
    """

    length = relative_angles.shape[0]
    distances = distance_signature(relative_angles)
    diff = 0
    for i in range(length):
        diff = max(diff, abs(distances[i]))
    return diff

def indent_score(relative_angles):
    """
        Compute score for indent part

        :param relative_angles: list of angles
        :return: List of floats
    """

    length = relative_angles.shape[0]
    peak = longest_peak(relative_angles)

    while peak[0] > 0 and not is_maximum_local(peak[0], relative_angles, 10):
        peak = (peak[0] - 1, peak[1])
    while peak[1] < length - 1 and not is_maximum_local(peak[1], relative_angles, 10):
        peak = (peak[0], peak[1] + 1)

    shape = np.zeros((peak[0] + length - peak[1], 1))
    for i in range(peak[0] + 1):
        shape[i] = relative_angles[i]
    for i in range(peak[1], length):
        shape[i - peak[1] + peak[0]] = relative_angles[i]

    # FIX FOR FUNCTIONS > 0
    if shape.shape[0] == 1:
        return flat_score(relative_angles)
    return flat_score(shape)

def outdent_score(relative_angles):
    """
        Compute score for outdent part

        :param relative_angles: list of angles
        :return: List of floats
    """

    return indent_score(-relative_angles)

def compute_comp(combs_l, relative_angles, method='correlate'):
    """
        Compute score for each combinations of 4 points and return the index of the best

        :param combs_l: list of combinations of 4 points
        :param relative_angles: List of angles
        :return: Int
    """

    # Combinations of 4 points
    global COUNT
    MY_COUNT = 0

    results_glob = []
    for comb_t in combs_l:

        # Roll the values of relative angles for this combination
        offset = len(relative_angles) - comb_t[3] - 1
        relative_angles_tmp = np.roll(relative_angles, offset)
        comb_t += offset
        comb_t = [(0, comb_t[0]), (comb_t[0], comb_t[1]), (comb_t[1], comb_t[2]), (comb_t[2], comb_t[3])]

        results_comp = []
        for comb in comb_t:

            hole, head, border = 0, 0, 0
            if method == 'flat':
                hole = indent_score(np.ravel(np.array(relative_angles_tmp[comb[0]:comb[1]])))
                head = outdent_score(np.ravel(np.array(relative_angles_tmp[comb[0]:comb[1]])))
                border = flat_score(np.ravel(np.array(relative_angles_tmp[comb[0]:comb[1]])))
        
            if hole != border:
                results_comp.append(np.min([hole, head]))
            else:
                results_comp.append(border)

        results_glob.append(np.sum(results_comp))
    return np.argmin(np.array(results_glob))

def peaks_inside(comb, peaks):
    """
        Check the number of peaks inside comb

        :param comb: Tuple of coordinates
        :param peaks: List of peaks to check
        :return: Int
    """

    cpt = []

    if len(comb) == 0:
        return cpt

    for peak in peaks:
        if peak > comb[0] and peak < comb[-1]:
            cpt.append(peak)
    return cpt

def is_pattern(comb, peaks):
    """
        Check if the peaks formed an outdent or an indent pattern

        :param comb: Tuple of coordinates
        :param peaks: List of peaks
        :return: Int
    """

    cpt = len(peaks_inside(comb, peaks))
    return cpt == 0 or cpt == 2 or cpt == 3

def is_acceptable_comb(combs, peaks, length):
    """
        Check if a combination is composed of acceptable patterns.
        Used to filter the obviously bad combinations quickly.

        :param comb: Tuple of coordinates
        :param peaks: List of peaks
        :param length: Length of the signature (used for offset computation)
        :return: Boolean
    """

    offset =  length - combs[3] - 1
    combs_tmp = combs + offset
    peaks_tmp = (peaks + offset) % length
    return is_pattern([0, combs_tmp[0]], peaks_tmp) and is_pattern([combs_tmp[0], combs_tmp[1]], peaks_tmp) and is_pattern([combs_tmp[1], combs_tmp[2]], peaks_tmp) and is_pattern([combs_tmp[2], combs_tmp[3]], peaks_tmp)

def type_peak(peaks_pos_inside, peaks_neg_inside):
    """
        Determine the type of lists of pos and neg peaks

        :param peaks_pos_inside: List of positive peaks
        :param peaks_neg_inside: List of negative peaks
        :return: TypeEdge
    """

    if len(peaks_pos_inside) == 0 and len(peaks_neg_inside) == 0:
        return TypeEdge.BORDER
    if len(peaks_inside(peaks_pos_inside, peaks_neg_inside)) == 2:
        return TypeEdge.HOLE
    if len(peaks_inside(peaks_neg_inside, peaks_pos_inside)) == 2:
        return TypeEdge.HEAD
    return TypeEdge.UNDEFINED

def my_find_corner_signature(cnt, green=False):
    """
        Determine the corner/edge positions by analyzing contours.

        :param cnt: contour to analyze
        :param green: boolean used to activate green background mode
        :type cnt: list of tuple of points
        :return: Corners coordinates, Edges lists of points, type of pieces
    """

    edges = []
    combs_final = []
    types_pieces = []
    sigma = 5
    max_sigma = 12
    if not green:
        sigma = 5
        max_sigma = 15
    while sigma <= max_sigma:
        print("Smooth curve with sigma={}...".format(sigma))

        tmp_combs_final = []

        # Find relative angles
        cnt_convert = [c[0] for c in cnt]
        relative_angles = get_relative_angles(np.array(cnt_convert), export=False, sigma=sigma)
        relative_angles = np.array(relative_angles)
        relative_angles_inverse = -np.array(relative_angles)
        
        extr_tmp = detect_peaks(relative_angles, mph=0.3*np.max(relative_angles))
        relative_angles = np.roll(relative_angles, int(len(relative_angles) / 2))
        extr_tmp = np.append(extr_tmp, (detect_peaks(relative_angles, mph=0.3*max(relative_angles)) - int(len(relative_angles) / 2)) % len(relative_angles), axis=0)
        relative_angles = np.roll(relative_angles, -int(len(relative_angles) / 2))
        extr_tmp = np.unique(extr_tmp)

        extr_tmp_inverse = detect_peaks(relative_angles_inverse, mph=0.3*np.max(relative_angles_inverse))
        relative_angles_inverse = np.roll(relative_angles_inverse, int(len(relative_angles_inverse) / 2))
        extr_tmp_inverse = np.append(extr_tmp_inverse, (detect_peaks(relative_angles_inverse, mph=0.3*max(relative_angles_inverse)) - int(len(relative_angles_inverse) / 2)) % len(relative_angles_inverse), axis=0)
        relative_angles_inverse = np.roll(relative_angles_inverse, -int(len(relative_angles_inverse) / 2))
        extr_tmp_inverse = np.unique(extr_tmp_inverse)

        extr = extr_tmp
        extr_inverse = extr_tmp_inverse

        relative_angles = sklearn.preprocessing.normalize(relative_angles[:,np.newaxis], axis=0).ravel()

        # Build list of permutations of 4 points
        combs = itertools.permutations(extr, 4)
        combs_l = list(combs)
        OFFSET_LOW = len(relative_angles) / 8
        OFFSET_HIGH = len(relative_angles) / 2.0
        for icomb, comb in enumerate(combs_l):
            if ((comb[0] > comb[1]) and (comb[1] > comb[2]) and (comb[2] > comb[3])
                and ((comb[0] - comb[1]) > OFFSET_LOW) and ((comb[0] - comb[1]) < OFFSET_HIGH)
                and ((comb[1] - comb[2]) > OFFSET_LOW) and ((comb[1] - comb[2]) < OFFSET_HIGH)
                and ((comb[2] - comb[3]) > OFFSET_LOW) and ((comb[2] - comb[3]) < OFFSET_HIGH)
                and ((comb[3] + (len(relative_angles) - comb[0])) > OFFSET_LOW) and ((comb[3] + (len(relative_angles) - comb[0])) < OFFSET_HIGH)):
                if is_acceptable_comb((comb[3], comb[2], comb[1], comb[0]), extr, len(relative_angles)) and is_acceptable_comb((comb[3], comb[2], comb[1], comb[0]), extr_inverse, len(relative_angles)):
                    tmp_combs_final.append((comb[3], comb[2], comb[1], comb[0]))
        sigma += 1
        if len(tmp_combs_final) == 0:
            continue

        best_fit = tmp_combs_final[compute_comp(tmp_combs_final, relative_angles, method='flat')]

        # Roll the values of relative angles for this combination
        offset = len(relative_angles) - best_fit[3] - 1
        relative_angles = np.roll(relative_angles, offset)
        best_fit += offset
        extr = (extr + offset) % len(relative_angles)
        extr_inverse = (extr_inverse + offset) % len(relative_angles)

        tmp_types_pieces = []
        no_undefined = True
        for best_comb in [[0, best_fit[0]], [best_fit[0], best_fit[1]], [best_fit[1], best_fit[2]], [best_fit[2], best_fit[3]]]:
            pos_peaks_inside = peaks_inside(best_comb, extr)
            neg_peaks_inside = peaks_inside(best_comb, extr_inverse)
            pos_peaks_inside.sort()
            neg_peaks_inside.sort()
            tmp_types_pieces.append(type_peak(pos_peaks_inside, neg_peaks_inside))
            if (tmp_types_pieces[-1] == TypeEdge.UNDEFINED):
                no_undefined = False
        
        combs_final = tmp_combs_final
        types_pieces = tmp_types_pieces

        if no_undefined:
            break
    
    if (len(types_pieces) != 0 and types_pieces[-1] == TypeEdge.UNDEFINED):
        print("UNDEFINED FOUND - try to continue but something bad happened :(")
        print(tmp_types_pieces[-1])

    best_fit_tmp = best_fit - offset
    for i in range(3):
        edges.append(cnt[best_fit_tmp[i]:best_fit_tmp[i + 1]])
    edges.append(np.concatenate((cnt[best_fit_tmp[3]:], cnt[:best_fit_tmp[0]]), axis=0))

    edges = [np.array([x[0] for x in e]) for e in edges]  # quick'n'dirty fix of the shape
    types_pieces.append(types_pieces[0])
    return best_fit, edges, types_pieces[1:]


def angle_between(v1, v2):
    """
        Return the angles between two tuples

        :param v1: first tuple of coordinates
        :param v2: second tuple of coordinates
        :return: distance Float
    """

    return math.atan2(-v1[1], v1[0]) - math.atan2(-v2[1], v2[0])

def export_contours(img, img_bw, contours, path, modulo, viewer=None, green=False):
    """
        Find the corners/shapes of all contours and build an array of puzzle Pieces

        :param img: matrix of the img
        :param img_bw: matrix of the img in black and white
        :param contours: lists of tuples of coordinates of contours
        :param path: Path used to export pieces img
        :path viewer: Object used for GUI display
        :param green: boolean used to activate green background mode
        :return: puzzle Piece array
    """

    puzzle_pieces = []
    list_img = []
    out_color = np.zeros_like(img)

    for idx, cnt in enumerate(contours):
        
        corners, edges_shape, types_edges = my_find_corner_signature(cnt, green)
        if corners is None:
            return None
        
        mask_border = np.zeros_like(img_bw)
        mask_full = np.zeros_like(img_bw)
        mask_full = cv2.drawContours(mask_full, contours, idx, 255, -1)
        mask_border = cv2.drawContours(mask_border, contours, idx, 255, 1)

        img_piece = np.zeros_like(img)
        img_piece[mask_full == 255] = img[mask_full == 255]

        pixels = []
        for x, y in tuple(zip(*np.where(mask_full == 255))):
            pixels.append(Pixel((x, y), img_piece[x, y]))

        color_vect = []

        # go faster, use only a subset of the img with the piece
        x_bound, y_bound, w_bound, h_bound = cv2.boundingRect(cnt)
        img_piece_tiny = img_piece[y_bound:y_bound + h_bound, x_bound:x_bound + w_bound]
        mask_border_tiny = mask_border[y_bound:y_bound + h_bound, x_bound:x_bound + w_bound]
        mask_full_tiny = mask_full[y_bound:y_bound + h_bound, x_bound:x_bound + w_bound]

        mask_around_tiny = np.zeros_like(mask_full_tiny)
        mask_inv_border_tiny = cv2.bitwise_not(mask_border_tiny)
        mask_full_tiny = cv2.bitwise_and(mask_full_tiny, mask_full_tiny, mask=mask_inv_border_tiny)

        for i in range(4):
            color_edge = []
            for ip, p in enumerate(edges_shape[i]):
                CIRCLE_SIZE = 5
                if ip != 0:
                    p2 = edges_shape[i][ip - 1]
                    cv2.circle(mask_around_tiny, (p2[0] - x_bound, p2[1] - y_bound), CIRCLE_SIZE, 0, -1)
                cv2.circle(mask_around_tiny, (p[0] - x_bound, p[1] - y_bound), CIRCLE_SIZE, 255, -1)

                mask_around_tiny = cv2.bitwise_and(mask_around_tiny, mask_around_tiny, mask=mask_full_tiny)

                neighbors_color = []
                for y, x in tuple(zip(*np.where(mask_around_tiny == 255))):
                    neighbors_color.append(img_piece_tiny[y, x])
                rgb = flatten_colors(neighbors_color)
                hsl = np.array(colorsys.rgb_to_hls(rgb[2] / 255.0, rgb[1] / 255.0, rgb[0] / 255.0))
                color_edge.append(hsl)
                out_color[p[1], p[0]] = rgb

            color_vect.append(np.array(color_edge))

        edges = []
        cpt = 0
        for s, c in zip(edges_shape, color_vect):
            edges.append(Edge(s, c, type=types_edges[cpt]))
            cpt += 1

        for i, e in enumerate(edges):
            e.direction = directions[i]
            if e.type == TypeEdge.BORDER:
                e.connected = True

        puzzle_pieces.append(PuzzlePiece(edges, pixels))

        mask_border = np.zeros_like(img_bw)

        for i in range(4):
            for p in edges_shape[i]:
                mask_border[p[1], p[0]] = 255

        out = np.zeros_like(img_bw)
        out[mask_border == 255] = img_bw[mask_border == 255]

        x, y, w, h = cv2.boundingRect(cnt)
        out2 = out[y:y + h, x:x + w]

        list_img.append(out2)

    max_height = max([x.shape[0] for x in list_img])
    max_width = max([x.shape[1] for x in list_img])
    pieces_img = np.zeros([max_height * (int(len(list_img) / modulo) + 1), max_width * modulo], dtype=np.uint8)
    for index, image in enumerate(list_img):
        pieces_img[(max_height * int(index / modulo)):(max_height * int(index / modulo) + image.shape[0]),
        (max_width * (index % modulo)):(max_width * (index % modulo) + image.shape[1])] = image


    cv2.imwrite("/tmp/color_border.png", out_color)
    cv2.imwrite(path, pieces_img)
    if viewer:
        viewer.addImage("Extracted colored border", "/tmp/color_border.png")

    return puzzle_pieces
