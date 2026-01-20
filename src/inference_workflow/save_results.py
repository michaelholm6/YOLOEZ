# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import sys
from PyQt5 import QtWidgets


def get_save_path(default_name="inference_results"):
    """
    Prompt the user to select a directory for saving inference results.
    
    Args:
        default_name (str): Default suggested directory name.
    
    Returns:
        str or None: The chosen directory path, or None if cancelled.
    """
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        owns_app = True

    dialog = QtWidgets.QFileDialog()
    dialog.setWindowTitle("Select Directory to Save Inference Results")
    dialog.setFileMode(QtWidgets.QFileDialog.Directory)
    dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
    dialog.selectFile(default_name)

    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        save_dir = dialog.selectedFiles()[0]
    else:
        save_dir = None

    if owns_app:
        app.quit()

    return save_dir