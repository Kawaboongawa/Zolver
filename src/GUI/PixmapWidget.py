import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QDesktopWidget
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, Qt

class PixmapWidget(QWidget):

    def __init__(self):
        super().__init__()
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
        self.horizontalGroupBox = QGroupBox("Grid")
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
