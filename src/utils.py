# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import cv2
import ctypes
from PyQt5 import QtWidgets, QtCore
import sys


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

    # -------- Sizing logic --------
    screen = QtWidgets.QApplication.primaryScreen()
    geom = screen.availableGeometry()

    dialog.setMinimumSize(450, 300)
    dialog.resize(min(700, int(geom.width() * 0.7)), min(500, int(geom.height() * 0.7)))

    result = dialog.exec_()

    # If dialog was closed with the window 'X', exit the script
    if result != QtWidgets.QDialog.Accepted:
        print("Instructions window closed — exiting script.")
        sys.exit(0)


def make_label_with_tooltip(text, tooltip):
    container = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    label = QtWidgets.QLabel(text)
    icon_label = QtWidgets.QLabel()
    icon_pix = QtWidgets.QApplication.style().standardIcon(
        QtWidgets.QStyle.SP_MessageBoxQuestion
    )
    icon_label.setPixmap(icon_pix.pixmap(20, 20))
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
