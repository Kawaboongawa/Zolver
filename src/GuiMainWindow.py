import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

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


    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '',"Image files (*.jpg *.png *.jpeg *.bmp)")
        if (fname[0]):
            kernel = np.ones((3, 3), np.uint8)
            gray_img = cv2.imread(fname[0], 0)
            rsz_img = cv2.resize(gray_img, None, fx=0.5, fy=0.5)
            blurred_img = cv2.GaussianBlur(rsz_img, (3, 3), 0)
            edges = self.auto_canny(blurred_img)
            dilated_img = cv2.dilate(edges, kernel, iterations=3)
            im2, contours, hier = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            puzzlepiece = Puzzle(contours)
            cv2.imwrite("/tmp/yolo.png", dilated_img);
            self.display_window_.display("/tmp/yolo.png")
            self.display_window_.show()

    display_window_ = None