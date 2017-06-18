from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from src.GuiDisplayWindow import GuiDisplayWindow


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
        fname = QFileDialog.\
            getOpenFileName(self, 'Open file', '',"Image files (*.jpg *.png *.jpeg *.bmp)")
        if fname[0]:
            self.display_window_.display(str(fname[0]))
            self.display_window_.show()

    display_window_ = None