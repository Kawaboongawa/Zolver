from colorsys import rgb_to_hls

import cv2
from scipy.signal import savgol_filter
import numpy as np
import scipy.misc
import imutils
import random
import math, pickle, os
import sklearn.preprocessing

from Img.Pixel import Pixel, flatten_colors
from Puzzle.Edge import Edge
from Puzzle.Enums import directions, TypeEdge
from Puzzle.PuzzlePiece import PuzzlePiece
from Puzzle.PuzzlePiece import normalize_edge, normalize_list
import matplotlib.pyplot as plt
import matplotlib
import scipy, sklearn.preprocessing
import itertools
from scipy.spatial.distance import euclidean, chebyshev
from Img.peak_detect import *

def auto_canny(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(img)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, 100, 200)
    # return the edged image
    return edges


# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg2 = cv2.createBackgroundSubtractorMOG2()
# fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG()

def edgedetect(channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255
    return sobel
    # Some values seem to go above 255. However RGB channels has to be within 0-255


def findSignificantContours(img, edgeImg):
    image, contours, heirarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)
    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = edgeImg.size * 5 / 100  # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:
        contour = contours[tupl[0]]
        area = cv2.contourArea(contour)
        if area > tooSmall:
            significant.append([contour, area])

            # Draw the contour on the original image
            cv2.drawContours(img, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

    significant.sort(key=lambda x: x[1])
    # print ([x[1] for x in significant]);
    return [x[0] for x in significant]


def findContourTest1(initial_img):
    edged = cv2.Canny(initial_img, 10, 250)
    cv2.imshow("Edges", edged)
    # applying closing function
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closed", closed)
    # finding_contours
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cv2.drawContours(initial_img, [approx], -1, (0, 255, 0), 2)
    cv2.imshow("Output", initial_img)


def find_corners(img):
    corners = cv2.goodFeaturesToTrack(img, 10, 0.001, 20, blockSize=20)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 10, 255, -1)
    return img


# Not working at all
def find_corners_mser(img):
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    img = cv2.polylines(img, hulls, 1, (0, 255, 0))
    return img


def my_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def clamp(a, threshmin=-0.1, threshmax=0.1):
    for ind, pt in enumerate(a):
        if pt < threshmin:
            a[ind] = -1
        elif pt > threshmax:
            a[ind] = 1
        else:
            a[ind] = 0


COUNT = 0


def get_relative_angles(cnt, export=False, norm=False):
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
    angles = scipy.ndimage.filters.gaussian_filter(angles, 10)

    angles = np.roll(np.array(angles), -length)
    angles = angles[0:length]
    
    # clamp(angles)
    # angles = sklearn.preprocessing.binarize(np.array(angles).reshape((len(cnt) - 1, 1)), threshold=0.1)

    if export:
        pickle.dump(angles, open("/tmp/save" + str(COUNT) + ".p", "wb"))

        plt.plot(np.append(angles, angles))
        plt.savefig("/tmp/fig" + str(COUNT) + ".png")
        plt.clf()
        plt.cla()
        plt.close()

    return angles


# Point farthest election
def my_find_corners(img, cnt):
    corners = []
    elect = [0] * len(cnt)
    for p in cnt:
        l = np.array([x[0] for x in cnt])
        res = my_dist(l, p)
        ind = np.argmax(res)
        elect[ind] += 1

    corners = []
    edges = []
    indices = []
    for i in range(4):
        ind = np.argmax(elect)
        value = cnt[ind][0]
        corners.append(value)
        indices.append(ind)
        elect[max(ind - 10, 0):min(ind + 10, len(elect))] = [0] * (min(ind + 10, len(elect)) - max(ind - 10, 0))
        # cv2.circle(img, tuple(value), 50, 255, -1)

    indices.sort()

    for i in range(3):
        edges.append(cnt[indices[i]:indices[i + 1]])
    edges.append(np.concatenate((cnt[indices[3]:], cnt[:indices[0]]), axis=0))

    edges = [np.array([x[0] for x in e]) for e in edges]  # quick'n'dirty fix of the shape
    return corners, edges

# Determine if a point at index is a maximum local in radius range of relative_angles function
def is_maximum_local(index, relative_angles, radius):
    start = max(0, index - radius)
    end = min(relative_angles.shape[0] - 1, index + radius)
    for i in range(start, end + 1):
        if relative_angles[i] > relative_angles[index]:
            return False
    return True

# Longest area < 0
def longest_peak(relative_angles):
    length = relative_angles.shape[0]
    longest = (0, 0)
    j = 0
    for i in range(length):
        if relative_angles[i] >= 0:
            j = i
        if i - j > longest[1] - longest[0]:
            longest = (j, i)
    return longest

# Distance of each points to the line formed by first and last points
def distance_signature(relative_angles):
    length = relative_angles.shape[0]
    
    l1 = np.array([0, relative_angles[0]])
    l2 = np.array([length - 1, relative_angles[-1]])
    
    '''
    #TESTS FOR LINE FLAT
    l1 = np.array([0, 0])
    l2 = np.array([length - 1, 0])
    '''
    signature = np.zeros((length, 1))

    for i in range(length):
        assert(np.linalg.norm(l2 - l1) != 0)
        signature[i] = np.linalg.norm(np.cross(l2 - l1, l1 - np.array([i, relative_angles[i]]))) / np.linalg.norm(l2 - l1)
    
    return signature

# Flat score
def flat_score(relative_angles):
    length = relative_angles.shape[0]
    distances = distance_signature(relative_angles)
    diff = 0
    for i in range(length):
        diff = max(diff, abs(distances[i]))
    return diff

# Compute score for indent part
def indent_score(relative_angles):
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

# Compute score for outdent part
def outdent_score(relative_angles):
    return indent_score(-relative_angles)

def compute_comp(combs_l, relative_angles, method='correlate'):
    # Combinations of 4 points
    # print("Number combinations: ", len(combs_l))
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
    cpt = []

    if len(comb) == 0:
        return cpt

    for peak in peaks:
        if peak > comb[0] and peak < comb[-1]:
            cpt.append(peak)
    return cpt

def is_pattern(comb, peaks):
    cpt = len(peaks_inside(comb, peaks))
    return cpt == 0 or cpt == 2 or cpt == 3

def is_acceptable_comb(combs, peaks, length):
    offset =  length - combs[3] - 1
    combs_tmp = combs + offset
    peaks_tmp = (peaks + offset) % length
    return is_pattern([0, combs_tmp[0]], peaks_tmp) and is_pattern([combs_tmp[0], combs_tmp[1]], peaks_tmp) and is_pattern([combs_tmp[1], combs_tmp[2]], peaks_tmp) and is_pattern([combs_tmp[2], combs_tmp[3]], peaks_tmp)

def type_peak(peaks_pos_inside, peaks_neg_inside):
    if len(peaks_pos_inside) == 0 and len(peaks_neg_inside) == 0:
        return TypeEdge.BORDER
    if len(peaks_inside(peaks_pos_inside, peaks_neg_inside)) == 2:
        return TypeEdge.HOLE
    if len(peaks_inside(peaks_neg_inside, peaks_pos_inside)) == 2:
        return TypeEdge.HEAD
    return TypeEdge.UNDEFINED

def my_find_corner_signature(img, cnt, piece_img=None):
    global COUNT
    COUNT = COUNT + 1

    corners = []
    edges = []

    # Find relative angles
    cnt_convert = [c[0] for c in cnt]
    relative_angles = get_relative_angles(np.array(cnt_convert), export=True)

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
    combs_final = []
    OFFSET_LOW = len(relative_angles) / 8
    OFFSET_HIGH = len(relative_angles) / 2
    for icomb, comb in enumerate(combs_l):
        if ((comb[0] > comb[1]) and (comb[1] > comb[2]) and (comb[2] > comb[3])
            and ((comb[0] - comb[1]) > OFFSET_LOW) and ((comb[0] - comb[1]) < OFFSET_HIGH)
            and ((comb[1] - comb[2]) > OFFSET_LOW) and ((comb[1] - comb[2]) < OFFSET_HIGH)
            and ((comb[2] - comb[3]) > OFFSET_LOW) and ((comb[2] - comb[3]) < OFFSET_HIGH)
            and ((comb[3] + (len(relative_angles) - comb[0])) > OFFSET_LOW) and ((comb[3] + (len(relative_angles) - comb[0])) < OFFSET_HIGH)):
            if is_acceptable_comb((comb[3], comb[2], comb[1], comb[0]), extr, len(relative_angles)) and is_acceptable_comb((comb[3], comb[2], comb[1], comb[0]), extr_inverse, len(relative_angles)):
                combs_final.append((comb[3], comb[2], comb[1], comb[0]))
            
    if len(combs_final) == 0:
        print("ERROR NO COMBINATIONS FOUND, exporting graph...")

        plt.figure(1)
        plt.subplot(211)

        for e in extr:
            plt.axvline(x=e, lw=0.2)

        for e in extr_inverse:
            plt.axvline(x=e, lw=0.2)
        
        plt.plot(relative_angles)
        ax=plt.gca()

        #plt.subplot(212)
        #plt.imshow(piece_img)
        #plt.axis("off")

        plt.savefig("/tmp/extr" + str(COUNT) + ".png", format='png', dpi=900)
        plt.clf()
        plt.cla()
        plt.close()
        exit(1)

    best_fit = combs_final[compute_comp(combs_final, relative_angles, method='flat')]

    # Roll the values of relative angles for this combination
    offset = len(relative_angles) - best_fit[3] - 1
    relative_angles = np.roll(relative_angles, offset)
    best_fit += offset
    extr = (extr + offset) % len(relative_angles)
    extr_inverse = (extr_inverse + offset) % len(relative_angles)

    types_pieces = []
    for best_comb in [[0, best_fit[0]], [best_fit[0], best_fit[1]], [best_fit[1], best_fit[2]], [best_fit[2], best_fit[3]]]:
        pos_peaks_inside = peaks_inside(best_comb, extr)
        neg_peaks_inside = peaks_inside(best_comb, extr_inverse)
        pos_peaks_inside.sort()
        neg_peaks_inside.sort()
        types_pieces.append(type_peak(pos_peaks_inside, neg_peaks_inside))
        if (types_pieces[-1] == TypeEdge.UNDEFINED):
            print("UNDEFINED FOUND - try to continue but something bad happened :(")
            print(types_pieces[-1])
            print(pos_peaks_inside)
            print(neg_peaks_inside)
    
    if piece_img is not None:
        plt.figure(1)
        plt.subplot(211)
        plt.axvline(x=0, lw=1, color='red')
        plt.axvline(x=best_fit[0], lw=1, color='red')
        plt.axvline(x=best_fit[1], lw=1, color='red')
        plt.axvline(x=best_fit[2], lw=1, color='red')
        plt.axvline(x=best_fit[3], lw=1, color='red')

        for e in extr:
            plt.axvline(x=e, lw=0.2)
        plt.plot(relative_angles)
        ax=plt.gca()
        #ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.75))

        plt.subplot(212)
        plt.imshow(piece_img)
        plt.axis("off")

        plt.savefig("/tmp/extr" + str(COUNT) + ".png", format='png', dpi=900)
        plt.clf()
        plt.cla()
        plt.close()

    best_fit_tmp = best_fit - offset
    for i in range(3):
        edges.append(cnt[best_fit_tmp[i]:best_fit_tmp[i + 1]])
    edges.append(np.concatenate((cnt[best_fit_tmp[3]:], cnt[:best_fit_tmp[0]]), axis=0))

    edges = [np.array([x[0] for x in e]) for e in edges]  # quick'n'dirty fix of the shape
    types_pieces.append(types_pieces[0])
    return best_fit, edges, types_pieces[1:]


def angle_between(v1, v2):
    return math.atan2(-v1[1], v1[0]) - math.atan2(-v2[1], v2[0])

# Return puzzle Piece array
def export_contours(img, img_bw, contours, path, modulo):
    puzzle_pieces = []
    list_img = []
    print('>>> START contour/corner detection')

    for idx, cnt in enumerate(contours):

        corners, edges_shape, types_edges = my_find_corner_signature(img_bw, cnt)
        #corners, edges_shape = my_find_corners(img_bw, cnt)

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

        out_color = np.zeros_like(img)
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
                hsl = np.array(rgb_to_hls(rgb[0], rgb[1], rgb[2]))
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
        cv2.imwrite("/tmp/color_border.png", out_color)

        mask_border = np.zeros_like(img_bw)

        for i in range(4):
            for p in edges_shape[i]:
                mask_border[p[1], p[0]] = 255

        out = np.zeros_like(img_bw)
        out[mask_border == 255] = img_bw[mask_border == 255]

        x, y, w, h = cv2.boundingRect(cnt)
        out2 = out[y:y + h, x:x + w]

        # NEED COMPUTE CENTER FROM 4 CORNERS (barycentre)
        # centerX = np.sum([x2[0] - x for x2 in corners]) / len(corners)
        # centerY = np.sum([x2[1] - y for x2 in corners]) / len(corners)
        # cv2.circle(out2, tuple((int(centerX), int(centerY))), 10, 255, -1)

        # /!\ /!\ TESTING PURPOSE ROTATE RANDOM AMOUNT /!\ /!\
        # out2 = imutils.rotate_bound(out2, random.randint(0, 360))

        # cv2.circle(out2, (int(w / 2), int(h / 2)), 10, 255, -1)

        # cv2.line(out2, tuple((int(centerX) - 1000, int(centerY) - 1000)), tuple((int(centerX) + 1000, int(centerY) + 1000)), 255, 5)
        # cv2.line(out2, tuple((corners[0][0] - x, corners[0][1] - y)), tuple((int(centerX), int(centerY))), 255, 5)

        # angle = np.degrees(angle_between((-1, 1, 0), (corners[0][0] - x - centerX, corners[0][1] - y - centerY, 0)))

        # print((h, h, 0), (corners[0][0] - x - centerX, corners[0][1] - y - centerY, 0), angle)

        #my_find_corner_signature(img_bw, cnt, out2)
        
        # rotated = imutils.rotate_bound(out2, angle)
        list_img.append(out2)

    # Normalize all edges to min edge
    #length = np.min([np.min([y.size for y in np.array(x.edges_)]) for x in puzzle_pieces])
    #for p in puzzle_pieces:
    #    p.normalize_edges(int(length / 3))

    max_height = max([x.shape[0] for x in list_img])
    max_width = max([x.shape[1] for x in list_img])
    pieces_img = np.zeros([max_height * (int(len(list_img) / modulo) + 1), max_width * modulo], dtype=np.uint8)
    for index, image in enumerate(list_img):
        pieces_img[(max_height * int(index / modulo)):(max_height * int(index / modulo) + image.shape[0]),
        (max_width * (index % modulo)):(max_width * (index % modulo) + image.shape[1])] = image

    cv2.imwrite(path, pieces_img)
    return puzzle_pieces


def load_signatures(path):
    holes, heads, borders = [], [], []
    holes = [x for x in os.listdir(path) if "hole" in x]
    heads = [x for x in os.listdir(path) if "head" in x]
    borders = [x for x in os.listdir(path) if "border" in x]

    holes = [pickle.load(open(os.path.join(path, f), "rb")) for f in holes]
    heads = [pickle.load(open(os.path.join(path, f), "rb")) for f in heads]
    borders = [pickle.load(open(os.path.join(path, f), "rb")) for f in borders]

    return {'holes': np.average(holes, axis=0), 'heads': np.average(heads, axis=0),
            'borders': np.average(borders, axis=0)}


def display(img, name='image'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift))
    return fshift, magnitude
