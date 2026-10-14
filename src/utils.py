# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import os
import sys

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore

_fallback_app = None


def read_image(path, flags=cv2.IMREAD_COLOR):
    """Read an image, returning None if it cannot be read.

    cv2.imread passes the path to the C++ layer, which on Windows cannot open
    anything outside the system code page, so a single accented or non-Latin
    character anywhere in the path makes it fail.  Reading the bytes in Python
    and decoding them in memory works for any path.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path, image):
    """Write an image, returning False if it could not be written.

    Encodes in memory and writes the bytes from Python for the same reason
    read_image reads them that way.
    """
    extension = os.path.splitext(path)[1]
    if not extension:
        extension = ".png"
        path = path + extension
    try:
        ok, buffer = cv2.imencode(extension, image)
    except cv2.error:
        ok = False
    if not ok:
        return False
    try:
        buffer.tofile(path)
    except OSError:
        return False
    return True


def ensure_app():
    """Return the running QApplication, creating one if there is none.

    The created application is kept in a module-level reference so it is not
    garbage collected while widgets are still using it.
    """
    global _fallback_app
    app = QtWidgets.QApplication.instance()
    if app is None:
        _fallback_app = QtWidgets.QApplication(sys.argv)
        app = _fallback_app
    return app


def show_instructions(message: str, title: str = "Instructions"):
    """Show a scrollable instructional dialog.  Exits the program if the user closes with X."""
    ensure_app()

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
    """Return a QWidget containing a text label and a question-mark icon with the given tooltip."""
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
    """Show a modal error message box."""
    ensure_app()

    msg_box = QtWidgets.QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setIcon(QtWidgets.QMessageBox.Critical)
    msg_box.setText(message)
    msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)

    msg_box.setMinimumWidth(400)
    msg_box.exec_()
