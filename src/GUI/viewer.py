#!/usr/bin/env python
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
                             QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QComboBox)

from Puzzle.Puzzle import Puzzle


class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.scaleFactor = 0.0

        self.currImg = 0
        self.imgs = []

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Zolver")
        self.resize(500, 400)

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            self.scaleFactor = 1.0

            self.imageLabel.adjustSize()
            self.zoomInAct.setEnabled(True)
            self.zoomOutAct.setEnabled(True)
            self.normalSizeAct.setEnabled(True)
            self.openAct.setEnabled(False)
            self.solveAct.setEnabled(True)
            self.addImage('Base image', fileName)

    def addImage(self, name, fileName, display=True):
        self.imgs.append(fileName)
        id = len(self.imgs) - 1
        self.imageMenu.addAction(QAction('&' + name, self, triggered=lambda: self.displayImage(id)))
        if display:
            self.displayImage(id)


    def displayImage(self, fileNameId):
        image = QImage(self.imgs[fileNameId])
        if image.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot load %s." % self.imgs[fileNameId])
            return
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.scaleFactor = 1.0
        self.imageLabel.adjustSize()
        self.currImg = fileNameId
        self.displayPrevAct.setEnabled(self.currImg != 0)
        self.displayNextAct.setEnabled(self.currImg + 1 != len(self.imgs))

    def displayNext(self):
        self.displayImage(self.currImg + 1)

    def displayPrev(self):
        self.displayImage(self.currImg - 1)

    def zoomIn(self):
        self.scaleImage(1.2)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def solve(self):
        self.solveAct.setEnabled(False)
        Puzzle(self.imgs[0], viewer=self)


    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                triggered=self.open)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl+P",
                enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+M",
                enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+N",
                enabled=False, triggered=self.normalSize)

        self.displayPrevAct = QAction("&Previous image", self, shortcut="Ctrl+A", enabled=False, triggered=self.displayPrev)

        self.displayNextAct = QAction("&Next image", self, shortcut="Ctrl+Z", enabled=False, triggered=self.displayNext)

        self.solveAct = QAction("&Solve puzzle", self, shortcut="Ctrl+S", enabled=False, triggered=self.solve)



    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.solveAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)

        self.imageMenu = QMenu("&Image", self)
        self.imageMenu.addAction(self.displayPrevAct)
        self.imageMenu.addAction(self.displayNextAct)
        self.imageMenu.addSeparator()


        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.imageMenu)


    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 10.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.01)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
