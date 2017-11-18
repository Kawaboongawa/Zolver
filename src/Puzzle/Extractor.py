from cv2 import cv2

import numpy as np
from Img.filters import *


class Extractor():
    def __init__(self, path, pixmapWidget=None):
        self.path = path
        self.img = cv2.imread(self.path, 1)
        self.pixmapWidget = pixmapWidget

    def extract(self):
        kernel = np.ones((3, 3), np.uint8)
        # img = cv2.resize(initial_img, None, fx=0.5, fy=0.5)

        cv2.imwrite("/tmp/yolo.png", self.img)
        if self.pixmapWidget is not None:
            self.pixmapWidget.add_image_widget("/tmp/yolo.png", 0, 0)

        # img = cv2.GaussianBlur(self.img, (3, 3), 0)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, self.img = cv2.threshold(self.img, 240, 255, cv2.THRESH_BINARY_INV)

        cv2.imwrite("/tmp/yolo.png", self.img)
        if self.pixmapWidget is not None:
            self.pixmapWidget.add_image_widget("/tmp/yolo.png", 1, 1)

        self.img, contours, hier = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        puzzle_pieces = export_contours(self.img, contours, "/tmp/contours.png", 5)
        if self.pixmapWidget is not None:
            self.pixmapWidget.add_image_widget("/tmp/contours.png", 0, 1)

        fshift, magnitude = get_fourier(self.img)
        cv2.imwrite("/tmp/yolo.png", magnitude)
        if self.pixmapWidget is not None:
            self.pixmapWidget.add_image_widget("/tmp/yolo.png", 1, 0)

        rows, cols = self.img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cv2.imwrite("/tmp/yolo.png", img_back)
        return puzzle_pieces
