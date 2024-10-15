import sys
from PyQt5.QtWidgets import QApplication
from GUI.Viewer import Viewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    imageViewer = Viewer()
    imageViewer.show()
    sys.exit(app.exec_())
