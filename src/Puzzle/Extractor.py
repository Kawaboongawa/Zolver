from cv2 import cv2
import sys
import numpy as np
from Img.filters import *


class Extractor():
    def __init__(self, path, pixmapWidget=None):
        self.path = path
        self.img = cv2.imread(self.path, 1)
        self.img_bw = cv2.imread(self.path, 1)
        self.pixmapWidget = pixmapWidget

    def extract(self):
        kernel = np.ones((3, 3), np.uint8)
        # img = cv2.resize(initial_img, None, fx=0.5, fy=0.5)

        cv2.imwrite("/tmp/binarized.png", self.img_bw)
        if self.pixmapWidget is not None:
            self.pixmapWidget.add_image_widget("/tmp/binarized.png", 0, 0)

        # img = cv2.GaussianBlur(self.img, (3, 3), 0)
        self.img_bw = cv2.cvtColor(self.img_bw, cv2.COLOR_BGR2GRAY)
        ret, self.img_bw = cv2.threshold(self.img_bw, 240, 255, cv2.THRESH_BINARY_INV)

        cv2.imwrite("/tmp/binarized_treshold.png", self.img_bw)



        def fill_holes():
            # filling contours found (and thus potentially holes in pieces)
            _, contour, _ = cv2.findContours(self.img_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                cv2.drawContours(self.img_bw, [cnt], 0, 255, -1)
            # Inversing colors : start of filling pieces
            self.img_bw = cv2.bitwise_not(self.img_bw)
            # filling holes in the pieces 2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)
            # Inversing colors back : end of filling pieces
            self.img_bw = cv2.bitwise_not(self.img_bw)

        cv2.imwrite("/tmp/binarized_treshold_filled.png", self.img_bw)
        if self.pixmapWidget is not None:
            self.pixmapWidget.add_image_widget("/tmp/binarized_treshold.png", 1, 1)

        # In case with fail to find the pieces, we fill some holes and then try again
        nb_error_max = 42
        while True:
            try:
                self.img_bw, contours, hier = cv2.findContours(self.img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                puzzle_pieces = export_contours(self.img, self.img_bw, contours, "/tmp/contours.png", 5)
                break
            except (IndexError):
                fill_holes()
                nb_error_max -= 1
                if nb_error_max <= 0:
                    print('Could not find the pieces, exiting the app')
                    sys.exit(1)
                print('Error while trying to find the pieces, trying again after filling some holes')
        if self.pixmapWidget is not None:
            self.pixmapWidget.add_image_widget("/tmp/contours.png", 0, 1)

        fshift, magnitude = get_fourier(self.img_bw)
        cv2.imwrite("/tmp/yolo.png", magnitude)
        if self.pixmapWidget is not None:
            self.pixmapWidget.add_image_widget("/tmp/yolo.png", 1, 0)

        rows, cols = self.img_bw.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cv2.imwrite("/tmp/yolo.png", img_back)
        return puzzle_pieces
