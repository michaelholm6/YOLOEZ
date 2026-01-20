# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

from PyQt5 import QtWidgets
import sys
import os
import shutil
from sklearn.model_selection import train_test_split
import tempfile
from utils import show_error_window

def split_dataset(folder, train_split=0.8):
    """
    Lets the user choose a raw dataset folder (or use the provided folder)
    and prepares it for YOLO training.
    """
    if folder is None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        folder = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Raw Dataset Folder")

    if folder and os.path.isdir(folder):
        # Output folder inside temp
        output_dir = os.path.join(tempfile.gettempdir(), "yolo_dataset")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare dataset
        prepare_yolo_dataset(folder, output_dir, train_split=train_split, class_names=["defect"])

        return output_dir  # path with YOLO-ready dataset + dataset.yaml
    else:
        return None


def prepare_yolo_dataset(input_dir, output_dir, train_split=0.8, class_names=None):
    """
    Prepares a YOLOv8-ready dataset structure by scanning all subdirectories for image+txt files.
    """
    if class_names is None:
        class_names = ["defect"]

    # Create output dirs
    images_train = os.path.join(output_dir, "images/train")
    images_val   = os.path.join(output_dir, "images/val")
    labels_train = os.path.join(output_dir, "labels/train")
    labels_val   = os.path.join(output_dir, "labels/val")
    for d in [images_train, images_val, labels_train, labels_val]:
        os.makedirs(d, exist_ok=True)

    # Collect image+label pairs
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]
    data_pairs = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in exts:
                img_path = os.path.join(root, f)
                label_path = os.path.join(root, name + ".txt")
                if os.path.exists(label_path):
                    data_pairs.append((img_path, label_path))

    if not data_pairs:
        show_error_window("No valid image-label pairs found in the selected dataset folder.")
        return None

    # Train/val split
    train_pairs, val_pairs = train_test_split(
        data_pairs, train_size=train_split, shuffle=True, random_state=42
    )

    # Copy files
    def copy_pairs(pairs, img_dest, label_dest):
        for img, lbl in pairs:
            shutil.copy(img, img_dest)
            shutil.copy(lbl, label_dest)

    copy_pairs(train_pairs, images_train, labels_train)
    copy_pairs(val_pairs, images_val, labels_val)

    # Write dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"names: {class_names}\n")

    return yaml_path
