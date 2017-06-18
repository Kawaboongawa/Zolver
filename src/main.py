import sys
from PyQt5.QtWidgets import QApplication, QWidget

from src.GuiDisplayWindow import GuiDisplayWindow
from src.GuiMainWindow import GuiMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dw = GuiDisplayWindow()
    main = GuiMainWindow(dw)
    sys.exit(app.exec_())