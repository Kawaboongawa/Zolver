import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from scipy.signal import savgol_filter

import numpy as np
from src.GuiDisplayWindow import GuiDisplayWindow
from src.Puzzle import *

class GuiMainWindow(QMainWindow):
    def __init__(self, display_window):
        super().__init__()
        self.display_window_ = display_window
        self.initUI()



    def initUI(self):
        display_window_ = GuiDisplayWindow()
        display_window_.show()
        self.textEdit = QTextEdit()
        self.statusBar()
        self.statusBar().showMessage('Ready')
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)


        openFile = QAction(QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        fileMenu.addAction(openFile)

        self.setGeometry(300, 300, 1980, 1080)
        self.setWindowTitle('zolver')
        self.show()

    def auto_canny(self, img, sigma=0.33):
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

    def edgedetect(self, channel):
        sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        sobel = np.hypot(sobelX, sobelY)

        sobel[sobel > 255] = 255
        return sobel
        # Some values seem to go above 255. However RGB channels has to be within 0-255

    def findSignificantContours(self, img, edgeImg):
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


    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '',"Image files (*.jpg *.png *.jpeg *.bmp)")
        if (fname[0]):
            kernel = np.ones((3, 3), np.uint8)
            initial_img = cv2.imread(fname[0], 1)
            rsz_img = cv2.resize(initial_img, None, fx=0.5, fy=0.5)

            # fgmask = self.fgbg.apply(rsz_img)
            # cv2.imshow('frame', fgmask)
            # fgmask2 = self.fgbg2.apply(rsz_img)
            # cv2.imshow('frame2', fgmask2)
            # fgmask3 = self.fgbg3.apply(rsz_img)
            # cv2.imshow('frame3', fgmask3)

            # self.findContourTest1(initial_img)

            # self.findContourTest2(initial_img)

            blurred_img = cv2.GaussianBlur(rsz_img, (3, 3), 0)
            edges = self.auto_canny(blurred_img)
            dilated_img = cv2.dilate(edges, kernel, iterations=3)
            im2, contours, hier = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            puzzlepiece = Puzzle(contours)

            res = initial_img
            cv2.imwrite("/tmp/yolo.png", res)
            self.display_window_.display("/tmp/yolo.png")
            self.display_window_.show()

    def findContourTest2(self, initial_img):
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
    def printImgContour(self, initial_img, contours):
        tmpimage = initial_img.copy()
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            cv2.drawContours(tmpimage, [approx], -1, (0, 255, 0), 2)
        self.imgNumber = self.imgNumber + 1
        print(str(self.imgNumber))
        cv2.imshow("imgContour" + str(self.imgNumber), tmpimage)
        cv2.waitKey(0)

    def findContourTest1(self, initial_img):
        edged = cv2.Canny(initial_img, 10, 250)
        cv2.imshow("Edges", edged)
        # applying closing function
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Closed", closed)
        # finding_contours
        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            cv2.drawContours(initial_img, [approx], -1, (0, 255, 0), 2)
        cv2.imshow("Output", initial_img)

    display_window_ = None