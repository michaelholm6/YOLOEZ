from PyQt5 import QtWidgets
import sys

def show_instructions(message: str, title: str = "Instructions"):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    msg_box = QtWidgets.QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setIcon(QtWidgets.QMessageBox.Information)
    msg_box.setText(message)  # use setText; QLabel inside handles wrapping automatically
    msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)

    # Optional: set a reasonable width
    msg_box.setMinimumWidth(400)  # avoid giant window
    msg_box.exec_()
