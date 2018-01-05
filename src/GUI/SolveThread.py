from PyQt5.QtCore import QThread

from Puzzle.Puzzle import Puzzle


class SolveThread(QThread):
    def __init__(self, path, viewer):
        QThread.__init__(self)
        self.path = path
        self.viewer = viewer

    def run(self):
        Puzzle(self.path, self.viewer)