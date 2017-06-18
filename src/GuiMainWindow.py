from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, qApp
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtWidgets import QPushButton, QMainWindow
from PyQt5.QtCore import QCoreApplication
import cv2
import matplotlib.pyplot as plt

class GuiMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.textEdit = QTextEdit()
        self.statusBar()
        self.statusBar().showMessage('Ready')
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
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

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('zolver')
        self.show()

    def showDialog(self):
        fname = QFileDialog.\
            getOpenFileName(self, 'Open file', '',"Image files (*.jpg *.png *.jpeg *.bmp)")
        if fname[0]:
            self.showimage(fname[0])

    def showimage(self, path):
        print(str(path))
        img = cv2.imread(str(path), 0)
        edge = cv2.Canny(img, 100, 200)
        plt.imshow(edge, cmap='gray')
        plt.show()

        #cv2.imshow('image', edge)
        #k = cv2.waitKey(0)
        #if k == 27:  # wait for ESC key to exit
            #cv2.destroyAllWindows()