# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import os
import cv2

def load_images_from_folder(folder_path, valid_exts=None):
    """
    Loads all images from a folder and returns them as a list of OpenCV BGR arrays.
    
    Args:
        folder_path (str): Path to the folder containing images.
        valid_exts (tuple, optional): Allowed image extensions. Defaults to common formats.
    
    Returns:
        List of OpenCV images (BGR numpy arrays)
    """
    if valid_exts is None:
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(valid_exts):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
    return images