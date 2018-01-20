from PyQt5.QtWidgets import QMessageBox, QScrollArea, QWidget, QVBoxLayout, QLabel


class ScrollMessageBox(QMessageBox):
    """ QMessageBox used to display the logs informations of the program """

    def __init__(self, l, *args, **kwargs):
        QMessageBox.__init__(self, *args, **kwargs)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        self.setWindowTitle("Zolver Logs")
        self.content = QWidget()
        scroll.setWidget(self.content)
        self.lay = QVBoxLayout(self.content)
        for item in l:
            self.lay.addWidget(QLabel(item, self))
        self.layout().addWidget(scroll, 0, 0, 1, self.layout().columnCount())
        self.setStyleSheet("QScrollArea{min-width:800 px; min-height: 600px}")