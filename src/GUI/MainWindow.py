from Puzzle import *
from GUI.PixmapWidget import *
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, qApp
from PyQt5.QtWidgets import QFileDialog

from GUI.PixmapWidget import PixmapWidget, QMainWindow
from Puzzle.Extractor import Extractor
from Puzzle.Puzzle import Puzzle


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
            p = Puzzle(fname[0], self.widget)
