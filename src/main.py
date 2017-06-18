import sys
from PyQt5.QtWidgets import QApplication, QWidget
from src.GuiMainWindow import GuiMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GuiMainWindow()
    sys.exit(app.exec_())