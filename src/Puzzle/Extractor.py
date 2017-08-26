import numpy as np
from Img.filters import *

class Extractor():
    def __init__(self, path, pixmapWidget):
        self.path = path
        self.pixmapWidget = pixmapWidget

    def extract(self):
        kernel = np.ones((3, 3), np.uint8)
        initial_img = cv2.imread(self.path, 1)
        rsz_img = cv2.resize(initial_img, None, fx=0.5, fy=0.5)

        cv2.imwrite("/tmp/yolo.png", initial_img)
        self.pixmapWidget.add_image_widget("/tmp/yolo.png", 0, 0)

        # fgmask = self.fgbg.apply(rsz_img)
        # cv2.imshow('frame', fgmask)
        # fgmask2 = self.fgbg2.apply(rsz_img)
        # cv2.imshow('frame2', fgmask2)
        # fgmask3 = self.fgbg3.apply(rsz_img)
        # cv2.imshow('frame3', fgmask3)

        # self.findContourTest1(initial_img)

        # self.findContourTest2(initial_img)

        img = cv2.GaussianBlur(rsz_img, (3, 3), 0)
        img = auto_canny(img)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        export_contours(img, contours, "/tmp/yolo.png", 5)
        self.pixmapWidget.add_image_widget("/tmp/yolo.png", 0, 1)

        fshift, magnitude = get_fourier(img)
        cv2.imwrite("/tmp/yolo.png", magnitude)
        self.pixmapWidget.add_image_widget("/tmp/yolo.png", 1, 0)

        rows, cols = img.shape
        crow, ccol = int(rows/2) , int(cols/2)
        fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cv2.imwrite("/tmp/yolo.png", img_back)
