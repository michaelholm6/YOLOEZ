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

def make_label_with_tooltip(text, tooltip):
    container = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    label = QtWidgets.QLabel(text)
    icon_label = QtWidgets.QLabel()
    icon_pix = QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
    icon_label.setPixmap(icon_pix.pixmap(14, 14))
    icon_label.setToolTip(tooltip)
    layout.addWidget(label)
    layout.addWidget(icon_label)
    layout.addStretch()
    return container