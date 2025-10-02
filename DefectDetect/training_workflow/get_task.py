from PyQt5 import QtWidgets, QtCore, QtGui

class TaskSelectionDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        font = QtGui.QFont()
        font.setPointSize(14)
        self.setFont(font)
        self.setWindowTitle("Select Task")

        # Center on screen
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2,
                  (screen.height() - self.height()) // 2)

        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("Please select a task:", self)
        layout.addWidget(label)

        self.segmentation_btn = QtWidgets.QPushButton("Segmentation", self)
        self.bbox_btn = QtWidgets.QPushButton("Bounding Box Creation", self)
        layout.addWidget(self.segmentation_btn)
        layout.addWidget(self.bbox_btn)

        self.chosen = None
        self.segmentation_btn.clicked.connect(lambda: self.set_choice("segmentation"))
        self.bbox_btn.clicked.connect(lambda: self.set_choice("detection"))

    def set_choice(self, choice):
        self.chosen = choice
        self.accept()  # closes the dialog

def get_task():
    # Check if QApplication exists; if not, create one
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if app is None:
        app = QtWidgets.QApplication([])
        owns_app = True

    dialog = TaskSelectionDialog()
    dialog.exec_()  # block until user makes a choice
    task = dialog.chosen

    # Only quit the app if we created it
    if owns_app:
        app.quit()

    return task