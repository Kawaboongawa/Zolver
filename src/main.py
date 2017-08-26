import sys
from PyQt5.QtWidgets import QApplication, QWidget

from GUI.GuiDisplayWindow import GuiDisplayWindow
from GUI.GuiMainWindow import GuiMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dw = GuiDisplayWindow()
    main = GuiMainWindow(dw)
    sys.exit(app.exec_())
