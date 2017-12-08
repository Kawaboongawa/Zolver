import cv2
from scipy.signal import savgol_filter
import numpy as np
from scipy.spatial import distance
import imutils
import random
import math, pickle, os

from Img.Pixel import Pixel, flatten_colors
from Puzzle.Edge import Edge
from Puzzle.Enums import directions, TypePiece
from Puzzle.PuzzlePiece import PuzzlePiece
from Puzzle.PuzzlePiece import normalize_edge, normalize_list
import matplotlib.pyplot as plt
import scipy, sklearn.preprocessing
import itertools


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

    angles = []
    last = np.pi
    for i in range(0, len(cnt) - 1):
        dir = (cnt[i + 1][0] - cnt[i][0], cnt[i + 1][1] - cnt[i][1])
        angle = math.atan2(-dir[1], dir[0])
        while (angle < last - np.pi):
            angle += 2 * np.pi
        while (angle > last + np.pi):
            angle -= 2 * np.pi
        angles.append(angle)
        last = angle

    angles = np.gradient(angles)
    angles = scipy.ndimage.filters.gaussian_filter(angles, 1)
    if norm:
        n = np.linalg.norm(angles)
        if n != 0:
            angles = angles / n

    # clamp(angles)
    # angles = sklearn.preprocessing.binarize(np.array(angles).reshape((len(cnt) - 1, 1)), threshold=0.1)

    if export:
        pickle.dump(angles, open("/tmp/save" + str(COUNT) + ".p", "wb"))

        plt.plot(angles)
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

def compute_comp(combs_l, relative_angles, signatures):
    # print('Number combs: ', len(combs_l))
    results_comp = []
    for comb in combs_l:
        if comb[1] - comb[0] < len(relative_angles) / 6 or comb[1] - comb[0] > len(relative_angles) / 3:
            results_comp.append([-1000000, -1000000, -1000000])
            continue

        tmp_relative = []
        tmp_signature = []
        if len(relative_angles[comb[0]:comb[1]]) < len(signatures['holes']):
            tmp_signature.append(np.array(normalize_list(signatures['holes'], len(relative_angles[comb[0]:comb[1]]))))
            tmp_relative.append(np.array(relative_angles[comb[0]:comb[1]]))
        else:
            tmp_relative.append(np.array(normalize_list(relative_angles[comb[0]:comb[1]], len(signatures['holes']))))
            tmp_signature.append(np.array(signatures['holes']))

        if len(relative_angles[comb[0]:comb[1]]) < len(signatures['heads']):
            tmp_signature.append(np.array(normalize_list(signatures['heads'], len(relative_angles[comb[0]:comb[1]]))))
            tmp_relative.append(np.array(relative_angles[comb[0]:comb[1]]))
        else:
            tmp_relative.append(np.array(normalize_list(relative_angles[comb[0]:comb[1]], len(signatures['heads']))))
            tmp_signature.append(np.array(signatures['heads']))

        if len(relative_angles[comb[0]:comb[1]]) < len(signatures['borders']):
            tmp_signature.append(np.array(normalize_list(signatures['borders'], len(relative_angles[comb[0]:comb[1]]))))
            tmp_relative.append(np.array(relative_angles[comb[0]:comb[1]]))
        else:
            tmp_relative.append(np.array(normalize_list(relative_angles[comb[0]:comb[1]], len(signatures['borders']))))
            tmp_signature.append(np.array(signatures['borders']))

        hole = np.correlate(tmp_relative[0], tmp_signature[0])
        head = np.correlate(tmp_relative[1], tmp_signature[1])
        border = np.correlate(tmp_relative[2], tmp_signature[2])
        results_comp.append([hole[0], head[0], border[0]])

    return results_comp

def my_find_corner_signature(img, cnt):
    global COUNT
    COUNT = COUNT + 1

    corners = []
    edges = []
    signatures = load_signatures("dataset")

    # Try smooh signatures
    signatures['holes'] = scipy.ndimage.filters.gaussian_filter(signatures['holes'], 2)
    signatures['heads'] = scipy.ndimage.filters.gaussian_filter(signatures['heads'], 2)
    signatures['borders'] = scipy.ndimage.filters.gaussian_filter(signatures['borders'], 2)

    # Find relative angles
    cnt_convert = [c[0] for c in cnt]
    cnt_convert = normalize_edge(cnt_convert, len(signatures['holes']) * 4)
    relative_angles = get_relative_angles(np.array(cnt_convert), export=False)

    # Introduce noise to find (flat peak flat) pattern
    noise = 1e-8 * np.asarray(random.sample(range(0, 1000), len(relative_angles)))
    relative_angles += noise

    # Find edges

    '''
    if len(signatures['holes']) > len(cnt_convert):
        signatures['holes'] = normalize_edge(signatures['holes'], len(cnt_convert))
        signatures['heads'] = normalize_edge(signatures['heads'], len(cnt_convert))
        signatures['borders'] = normalize_edge(signatures['borders'], len(cnt_convert))
    elif len(signatures['holes']) < len(cnt_convert):
        cnt_convert = normalize_edge(cnt_convert, len(signatures['holes']))
    '''

    extr_tmp = scipy.signal.argrelextrema(relative_angles, np.greater, mode='wrap', order=1)

    # Keep only 20% max values
    s = np.flip(np.argsort([relative_angles[x] for x in extr_tmp]), axis=1)[0]
    extr = []
    for i in range(int(len(extr_tmp[0]) * 0.2)):
        extr.append(extr_tmp[0][s[i]])

    n = np.linalg.norm(relative_angles)
    if n != 0:
        relative_angles = relative_angles / n

    n = np.linalg.norm(signatures['holes'])
    if n != 0:
        signatures['holes'] = signatures['holes'] / n

    n = np.linalg.norm(signatures['heads'])
    if n != 0:
        signatures['heads'] = signatures['heads'] / n

    n = np.linalg.norm(signatures['borders'])
    if n != 0:
        signatures['borders'] = signatures['borders'] / n

    combs = itertools.combinations(extr, 2)
    combs_l = list(combs)
    for icomb, comb in enumerate(combs_l):
        if comb[0] > comb[1]:
            combs_l[icomb] = (comb[1], comb[0])

    # print('Number combs: ', len(combs_l))
    results_comp = compute_comp(combs_l, relative_angles, signatures)

    index_max_hole = np.argmax([x[0] for x in results_comp])
    index_max_head = np.argmax([x[1] for x in results_comp])
    index_max_border = np.argmax([x[2] for x in results_comp])

    to_remove = []
    t = np.argmax([results_comp[index_max_hole][0], results_comp[index_max_head][1], results_comp[index_max_border][2]])
    # print(t)

    index = None
    if t == 0:
        index = index_max_hole
    elif t == 1:
        index = index_max_head
    elif t == 2:
        index = index_max_border

    # Roll values
    offset = len(relative_angles) - combs_l[index][1] - 1
    relative_angles = np.roll(relative_angles, offset)

    # Remove extremums between already found edge
    extr_tmp = []
    for ie, e in enumerate(extr):
        if e > combs_l[index][0] and e <= combs_l[index][1]:
            continue
        extr_tmp.append(e)
    extr = np.array(extr_tmp)

    for ie, e in enumerate(extr):
        extr[ie] = (extr[ie] + offset) % len(relative_angles)
    extr = np.append(extr, 0)

    plt.axvline(x=np.max(extr), lw=1, color='red')
    plt.axvline(x=len(relative_angles) - 1, lw=1, color='red')

    combs = itertools.combinations(extr, 2)
    combs_l = list(combs)
    for icomb, comb in enumerate(combs_l):
        if comb[0] > comb[1]:
            combs_l[icomb] = (comb[1], comb[0])

    results_comp = compute_comp(combs_l, relative_angles, signatures)

    for i in range(1, 4):
        if len(results_comp) == 0:
            break

        max_left = np.max([x[1] for x in combs_l])

        if i == 3:
            plt.axvline(x=0, lw=1, color='red')
            plt.axvline(x=max_left, lw=1, color='red')
            # TODO: Find type of edge
            continue

        index_max_hole = np.argmax(
            [x[0] if combs_l[ix][1] == max_left else -100000 for ix, x in enumerate(results_comp)])
        index_max_head = np.argmax(
            [x[1] if combs_l[ix][1] == max_left else -100000 for ix, x in enumerate(results_comp)])
        index_max_border = np.argmax(
            [x[2] if combs_l[ix][1] == max_left else -100000 for ix, x in enumerate(results_comp)])

        to_remove = []
        t = np.argmax(
            [results_comp[index_max_hole][0], results_comp[index_max_head][1], results_comp[index_max_border][2]])
        if t == 0:

            # print(combs_l[index_max_hole])
            plt.axvline(x=combs_l[index_max_hole][0], lw=1, color='red')
            plt.axvline(x=combs_l[index_max_hole][1], lw=1, color='red')

            for ic, c in enumerate(combs_l):
                if (c[0] <= combs_l[index_max_hole][0] and c[1] <= combs_l[index_max_hole][0]) or (
                        c[0] >= combs_l[index_max_hole][1] and c[1] >= combs_l[index_max_hole][1]):
                    continue
                to_remove.append(ic)

        elif t == 1:

            # print(combs_l[index_max_head])
            plt.axvline(x=combs_l[index_max_head][0], lw=1, color='red')
            plt.axvline(x=combs_l[index_max_head][1], lw=1, color='red')

            # Need remove all combs inside
            for ic, c in enumerate(combs_l):
                if (c[0] <= combs_l[index_max_head][0] and c[1] <= combs_l[index_max_head][0]) or (
                        c[0] >= combs_l[index_max_head][1] and c[1] >= combs_l[index_max_head][1]):
                    continue
                to_remove.append(ic)

        elif t == 2:

            # print(combs_l[index_max_border])
            plt.axvline(x=combs_l[index_max_border][0], lw=1, color='red')
            plt.axvline(x=combs_l[index_max_border][1], lw=1, color='red')

            # Need remove all combs inside
            for ic, c in enumerate(combs_l):
                if (c[0] <= combs_l[index_max_border][0] and c[1] <= combs_l[index_max_border][0]) or (
                        c[0] >= combs_l[index_max_border][1] and c[1] >= combs_l[index_max_border][1]):
                    continue
                to_remove.append(ic)

        for ic in sorted(to_remove, reverse=True):
            del combs_l[ic]
            del results_comp[ic]

            # print("/tmp/extr" + str(COUNT) + ".png: ", 'index ', i, ' is a: ', t)

    for e in extr:
        plt.axvline(x=e, lw=0.2)
    plt.plot(relative_angles)
    # plt.savefig("/tmp/extr" + str(COUNT) + ".png", format='png')
    plt.clf()
    plt.cla()
    plt.close()

    return corners, edges


def angle_between(v1, v2):
    if v1 == v2:
        return 0
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# Return puzzle Piece array
def export_contours(img, img_bw, contours, path, modulo):
    puzzle_pieces = []
    list_img = []
    print('>>> START contour/corner detection')

    for idx, cnt in enumerate(contours):

        my_find_corner_signature(img_bw, cnt)
        corners, edges_shape = my_find_corners(img_bw, cnt)

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
                color_edge.append(np.array(flatten_colors(neighbors_color)))
                out_color[p[1], p[0]] = color_edge[-1]
            color_vect.append(np.array(color_edge))

        edges = []
        for s, c in zip(edges_shape, color_vect):
            edges.append(Edge(s, c))

        for i, e in enumerate(edges):
            e.direction = directions[i]
            if e.is_border(1000):
                e.connected = True
                e.type = TypePiece.BORDER

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
