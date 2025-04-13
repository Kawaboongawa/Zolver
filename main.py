import atexit
import os
import sys
import tempfile

from PyQt5.QtWidgets import QApplication

from GUI.Viewer import Viewer


if __name__ == "__main__":
    # Create and use temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    os.environ["ZOLVER_TEMP_DIR"] = temp_dir.name
    atexit.register(temp_dir.cleanup)

    # Display GUI and exit
    app = QApplication(sys.argv)
    imageViewer = Viewer()
    imageViewer.show()
    sys.exit(app.exec_())
