# train_test_split.py
from PyQt5 import QtWidgets, QtCore, QtGui
import sys

def get_train_test_split():

    dialog = QtWidgets.QInputDialog()
    font = QtGui.QFont()
    font.setPointSize(14)  # increase this value as needed
    dialog.setFont(font)
    dialog.setWindowTitle("Train/Test Split")
    dialog.setLabelText("Enter train split percentage (0–100):")
    dialog.setInputMode(QtWidgets.QInputDialog.DoubleInput)
    dialog.setDoubleRange(10.0, 90.0)
    dialog.setDoubleDecimals(1)
    dialog.setDoubleValue(80.0)

    # Make the dialog appear in front initially
    dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
    dialog.show()
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
    dialog.show()

    # Show dialog modally
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        return dialog.doubleValue() / 100.0

    return None

if __name__ == "__main__":
    split = get_train_test_split()
    print("Train split:", split)