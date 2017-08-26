import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from scipy.signal import savgol_filter

import numpy as np
from GUI.GuiDisplayWindow import GuiDisplayWindow
from Puzzle.Puzzle import *
from Img.filters import *

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

            img = cv2.GaussianBlur(rsz_img, (3, 3), 0)
            img = auto_canny(img)
            img = cv2.dilate(img, kernel, iterations=1)
            img = cv2.erode(img, kernel, iterations=1)
            img, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            export_contours(img, contours, "/tmp/yolo.png")

            self.display_window_.display("/tmp/yolo.png")
            self.display_window_.show()

    display_window_ = None
