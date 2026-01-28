# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

from PyQt5 import QtWidgets
import sys
import os


def get_model_path():
    """
    Opens a file dialog for the user to select a trained YOLO model (.pt or .pth).

    Returns:
        str: Path to the selected model file, or None if cancelled.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    file_dialog = QtWidgets.QFileDialog()
    file_dialog.setWindowTitle("Select Trained YOLO Model")
    file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    file_dialog.setNameFilters(["YOLO Models (*.pt *.pth)"])

    if file_dialog.exec_():
        selected_files = file_dialog.selectedFiles()
        if selected_files:
            model_path = selected_files[0]
            if os.path.exists(model_path):
                return model_path
            else:
                print(f"File does not exist: {model_path}")
                return None
    return None
