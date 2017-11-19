import cv2
from scipy.signal import savgol_filter
import numpy as np
from scipy.spatial import distance
import imutils
import random
import math

from Puzzle.PuzzlePiece import PuzzlePiece


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


def findContourTest2(initial_img):
    blurred = cv2.GaussianBlur(initial_img, (5, 5), 0)  # Remove noise
    edgeImg = np.max(
        np.array([self.edgedetect(blurred[:, :, 0]),
                  self.edgedetect(blurred[:, :, 1]),
                  self.edgedetect(blurred[:, :, 2])]),
        axis=0)
    cv2.imshow("frame1", edgeImg)
    cv2.waitKey(0)
    mean = np.mean(edgeImg)
    # Zero any value that is less than mean. This reduces a lot of noise.
    edgeImg[edgeImg <= mean] = 0
    cv2.imshow("frame2", edgeImg)
    cv2.waitKey(0)
    edgeImg_8u = np.asarray(edgeImg, np.uint8)
    # Find contours
    significant = self.findSignificantContours(initial_img, edgeImg_8u)
    self.printImgContour(initial_img, significant)
    # Mask
    mask = edgeImg.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    # Invert mask
    mask = np.logical_not(mask)
    # Finally remove the background
    initial_img[mask] = 0
    self.printImgContour(initial_img, significant)
    contour = significant
    epsilon = 0.10 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 3, True)
    contour = approx
    # Use Savitzky-Golay filter to smoothen contour.
    # Consider each window to be 5% of image dimensions
    window_size = int(
        round(min(initial_img.shape[0], initial_img.shape[1]) * 0.05))
    x = savgol_filter(contour[:, 0, 0], window_size * 2 + 1, 3)
    y = savgol_filter(contour[:, 0, 1], window_size * 2 + 1, 3)
    approx = np.empty((x.size, 1, 2))
    approx[:, 0, 0] = x
    approx[:, 0, 1] = y
    approx = approx.astype(int)
    contour = approx
    self.printImgContour(initial_img, contour)


imgNumber = 0


def printImgContour(initial_img, contours):
    tmpimage = initial_img.copy()
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cv2.drawContours(tmpimage, [approx], -1, (0, 255, 0), 2)
    self.imgNumber = self.imgNumber + 1
    print(str(self.imgNumber))
    cv2.imshow("imgContour" + str(self.imgNumber), tmpimage)
    cv2.waitKey(0)


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

    return corners, edges


def angle_between(v1, v2):
    if v1 == v2:
        return 0
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# Return puzzle Piece array
def export_contours(img, img_bw, contours, path, modulo):
    puzzle_pieces = []
    list_img = []

    for idx, cnt in enumerate(contours):
        corners, edges = my_find_corners(img_bw, cnt)
        mask_border = np.zeros_like(img_bw)
        mask_full = np.zeros_like(img_bw)
        mask_full = cv2.drawContours(mask_full, contours, idx, 255, -1)

        img_piece = np.zeros_like(img)
        img_piece[mask_full == 255] = img[mask_full == 255]

        x, y, w, h = cv2.boundingRect(cnt)
        img_piece_crop = img_piece[y:y + h, x:x + w]
        puzzle_pieces.append(PuzzlePiece(edges, img_piece_crop))



        # cv2.imshow('image', img_piece_crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        for i in range(4):
            for p in edges[i]:
                mask_border[p[0][1], p[0][0]] = 255

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

    max_height = max([x.shape[0] for x in list_img])
    max_width = max([x.shape[1] for x in list_img])
    pieces_img = np.zeros([max_height * (int(len(list_img) / modulo) + 1), max_width * modulo], dtype=np.uint8)
    for index, image in enumerate(list_img):
        pieces_img[(max_height * int(index / modulo)):(max_height * int(index / modulo) + image.shape[0]),
        (max_width * (index % modulo)):(max_width * (index % modulo) + image.shape[1])] = image

    cv2.imwrite(path, pieces_img)
    return puzzle_pieces


def get_fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift))
    return fshift, magnitude
