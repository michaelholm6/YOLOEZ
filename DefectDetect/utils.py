from PyQt5 import QtWidgets
import sys

def show_instructions(message: str, title: str = "Instructions"):
    """
    Show a blocking popup with instructions for the user.
    Execution will pause until the user closes the popup.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    msg_box = QtWidgets.QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setIcon(QtWidgets.QMessageBox.Information)
    msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)

    # Put instructions into a QLabel so we can control sizing
    label = QtWidgets.QLabel(message)
    label.setWordWrap(True)  # allows wrapping instead of endless width
    label.setMinimumWidth(800)  # starting width
    label.setMinimumHeight(200) # starting height
    msg_box.layout().addWidget(label, 0, 1)  # add label to message box layout

    msg_box.exec_()