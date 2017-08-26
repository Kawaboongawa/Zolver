import sys
from PyQt5.QtWidgets import QApplication, QWidget
from GUI.MainWindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # dw = GuiDisplayWindow()

    # main = GuiMainWindow()
    widget = MainWindow()
    widget.show()

    sys.exit(app.exec_())
