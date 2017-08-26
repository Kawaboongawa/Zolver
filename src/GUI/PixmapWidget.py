import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, Qt
from Puzzle.Extractor import *

class MainWindow(QMainWindow):
    def __init__(self, parent=None):

        super(MainWindow, self).__init__(parent)
        self.widget = PixmapWidget(self)
        self.setCentralWidget(self.widget)
        self.initMenuBar()

    def initMenuBar(self):
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

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '',"Image files (*.jpg *.png *.jpeg *.bmp)")
        if (fname[0]):
            e = Extractor(fname[0], self.widget)
            e.extract()

class PixmapWidget(QWidget):

    def __init__(self, parent):
        super(PixmapWidget, self).__init__(parent)
        self.title = 'Image'
        self.left = 10
        self.top = 10
        self.width = QDesktopWidget().screenGeometry(-1).width()
        self.height = QDesktopWidget().screenGeometry(-1).height()
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.createGridLayout()

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.horizontalGroupBox)
        self.setLayout(windowLayout)

    def createGridLayout(self):
        self.horizontalGroupBox = QGroupBox("Zolver")
        layout = QGridLayout()
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 2)
        self.horizontalGroupBox.setLayout(layout)

    def remove_widget(self, index):
        item = self.horizontalGroupBox.layout().takeAt(index)
        if item:
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def add_image_widget(self, path, row, column):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(self.width / 2.1, self.height / 2.1, Qt.KeepAspectRatio)

        layout = self.horizontalGroupBox.layout()
        label = QLabel(self)
        label.setPixmap(pixmap)
        self.horizontalGroupBox.layout().addWidget(label, row, column)
