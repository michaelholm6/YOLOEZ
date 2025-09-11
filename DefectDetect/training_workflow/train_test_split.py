# train_test_split.py
from PyQt5 import QtWidgets
import sys

def get_train_test_split():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    val, ok = QtWidgets.QInputDialog.getDouble(
        None,
        "Train/Test Split",
        "Enter train split percentage (0–100):",
        80.0, 10.0, 90.0, 1
    )
    if ok:
        return val / 100.0  # return fraction
    return None

if __name__ == "__main__":
    split = get_train_test_split()
    print("Train split:", split)