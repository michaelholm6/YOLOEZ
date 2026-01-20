# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

from PyQt5 import QtWidgets
import sys
import os

def choose_image_folder():
    """
    Opens a folder selection dialog to let the user choose a directory
    containing images for inference.

    Returns:
        List of image file paths within the selected folder.
        Returns an empty list if no folder or no images are found.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Open folder dialog
    folder = QtWidgets.QFileDialog.getExistingDirectory(
        None,
        "Select Folder Containing Images for Inference",
        "",
        QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
    )

    if not folder:
        return []  # user cancelled

    # Gather supported image files from the folder
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    image_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]

    if not image_files:
        print("⚠ No image files found in the selected folder.")

    return image_files
