import cv2
import ctypes
import tkinter as tk
from tkinter import messagebox
from PyQt5 import QtWidgets, QtGui, QtCore
import sys

def resize_for_display(img, max_width, max_height):
    h, w = img.shape[:2]
    scale = 1.0
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img, scale

def get_screen_size(scale=0.9):
    user32 = ctypes.windll.user32
    screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return int(screen_width * scale), int(screen_height * scale)

def show_instructions(message: str, title: str = "Instructions"):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle(title)
    dialog.setModal(True)

    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(12, 12, 12, 12)

    # Scrollable text area
    label = QtWidgets.QLabel(message)
    label.setWordWrap(True)
    label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setWidget(label)

    layout.addWidget(scroll)

    # OK button
    btn = QtWidgets.QPushButton("OK")
    btn.clicked.connect(dialog.accept)
    btn.setDefault(True)

    btn_layout = QtWidgets.QHBoxLayout()
    btn_layout.addStretch()
    btn_layout.addWidget(btn)
    layout.addLayout(btn_layout)

    # -------- Sizing logic (key part) --------
    screen = QtWidgets.QApplication.primaryScreen()
    geom = screen.availableGeometry()

    dialog.setMinimumSize(450, 300)
    dialog.resize(
        min(700, int(geom.width() * 0.7)),
        min(500, int(geom.height() * 0.7))
    )

    dialog.exec_()

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

def show_error_window(message: str, title: str = "Error"):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    msg_box = QtWidgets.QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setIcon(QtWidgets.QMessageBox.Critical)
    msg_box.setText(message)
    msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)

    msg_box.setMinimumWidth(400) 
    msg_box.exec_()