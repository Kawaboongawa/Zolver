from PyQt5.QtCore import QDir
from PyQt5.QtGui import QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QFileDialog, QLabel,
                             QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)

from GUI.ScrollMessageBox import ScrollMessageBox
from GUI.SolveThread import SolveThread


class Viewer(QMainWindow):
    """ Main viewer window """

    def __init__(self):
        super(Viewer, self).__init__()

        self.scaleFactor = 1.0

        self.currImg = 0
        self.imgs = []
        self.img_names = []

        self.logs = []

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
        '''
        Callback for image opening
        '''
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        if fileName:
            self.imageLabel.adjustSize()
            self.zoomInAct.setEnabled(True)
            self.zoomOutAct.setEnabled(True)
            self.normalSizeAct.setEnabled(True)
            self.openAct.setEnabled(False)

            self.solveAct.setEnabled(True)
            self.solveGreenAct.setEnabled(True)

            self.addImage('Base image', fileName, addMenu=True)

    def addImage(self, name, fileName, display=True, addMenu=False):
        '''
        Add an image to the list of image

        :param name: The displayed name
        :param fileName: The path to the image
        :param display: Change the current image to the added one
        :param addMenu: Add the image directly to the menu
        '''
        self.imgs.append(fileName)
        self.img_names.append(name)
        id = len(self.imgs) - 1
        if addMenu:
            self.imageMenu.addAction(QAction('&' + name, self, triggered=lambda: self.displayImage(id)))
        if display:
            self.displayImage(id)

    def addLog(self, args):
        '''
        Add a phrase to the logs
        :param args: The new log line
        :return:
        '''
        self.logs.append(' '.join(map(str, args)))

    def displayImage(self, fileNameId):
        '''
        Display an image in the GUI
        :param fileNameId: The image ID
        '''
        image = QImage(self.imgs[fileNameId])
        if image.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot load %s." % self.imgs[fileNameId])
            return
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.scaleImage(1)
        #self.imageLabel.adjustSize()
        self.currImg = fileNameId
        self.displayPrevAct.setEnabled(self.currImg != 0)
        self.displayNextAct.setEnabled(self.currImg + 1 != len(self.imgs))

    def displayNext(self):
        '''
        Callback to display the next image
        '''
        self.displayImage(self.currImg + 1)

    def displayPrev(self):
        '''
        Callback to display the previous image
        '''
        self.displayImage(self.currImg - 1)

    def zoomIn(self):
        '''
        Callback to zoom in in the image
        '''
        self.scaleImage(1.2)

    def zoomOut(self):
        '''
        Callback to zoom out in the image
        '''
        self.scaleImage(0.8)

    def normalSize(self):
        '''
        Callback to reset the zoom of the image
        '''
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def solve(self):
        '''
        Callback to start resolution
        '''
        self.solveAct.setEnabled(False)
        self.solveGreenAct.setEnabled(False)

        self.solveMenu = QMenu("&Zolver is running", self)
        self.menuBar().addMenu(self.solveMenu)

        self.thread = SolveThread(self.imgs[0], self)
        self.thread.finished.connect(self.endSolve)
        self.thread.start()

    def solveGreen(self):
        '''
        Callback to start resolution using green background preprocessing
        '''
        self.solveAct.setEnabled(False)
        self.solveGreenAct.setEnabled(False)

        self.solveMenu = QMenu("&Zolver is running", self)
        self.menuBar().addMenu(self.solveMenu)

        self.thread = SolveThread(self.imgs[0], self, green_screen=True)
        self.thread.finished.connect(self.endSolve)
        self.thread.start()

    def endSolve(self):
        '''
        Callback at the end of the solving
        '''
        for id, n in enumerate(self.img_names):
            if id == 0:
                continue
            self.addOption(n, id)
        self.solveMenu.setEnabled(False)

    def addOption(self, n, id):
        '''
        Add an option to the menu
        :param n: Name
        :param id: Image id
        '''
        self.imageMenu.addAction(QAction('&' + n, self, triggered=lambda: self.displayImage(id)))

    def showLogs(self):
        '''
        Display the log window
        '''
        self.logWindow = ScrollMessageBox((str(x) for x in self.logs))
        self.logWindow.exec_()

    def createActions(self):
        '''
        Bind button with callbacks
        '''
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Up", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Down", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+N", enabled=False, triggered=self.normalSize)
        self.displayPrevAct = QAction("&Previous image", self, shortcut="Left", enabled=False, triggered=self.displayPrev)
        self.displayNextAct = QAction("&Next image", self, shortcut="Right", enabled=False, triggered=self.displayNext)
        self.solveAct = QAction("&Solve puzzle", self, shortcut="Ctrl+S", enabled=False, triggered=self.solve)
        self.solveGreenAct = QAction("&Solve puzzle (Green Background)", self, shortcut="Alt+S", enabled=False, triggered=self.solveGreen)
        self.logsAct = QAction("&Logs", self, shortcut="Ctrl+L", triggered=self.showLogs)

    def createMenus(self):
        '''
        Create all the GUI buttons
        '''
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.solveAct)
        self.fileMenu.addAction(self.solveGreenAct)
        self.fileMenu.addAction(self.logsAct)
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
        '''
        Scale the image with a factor
        :param factor: The scale factor
        '''
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 10.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.01)

    def adjustScrollBar(self, scrollBar, factor):
        '''
        Adjust the scrollbar size
        :param scrollBar: The widget
        :param factor: The factor
        :return:
        '''
        scrollBar.setValue(int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep()/2)))
