from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt


class GuiDisplayWindow(QLabel):
    def __init__(self):
        super(GuiDisplayWindow, self).__init__()
        self.initUI()


    def initUI(self):
        self.setWindowTitle('image')
        self.setGeometry(300, 300, 350, 100)

    def display(self, path):
        print(path)
        myPixmap = QPixmap(str(path))
        pixmap_resized = myPixmap.scaled(1080, 720, Qt.KeepAspectRatio)
        self.setPixmap(pixmap_resized)
        self.show()
