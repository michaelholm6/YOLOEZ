# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

from PyQt5 import QtWidgets
import os
import shutil
from sklearn.model_selection import train_test_split
import tempfile
from utils import ensure_app, show_error_window


def split_dataset(folder, train_split=0.8, image_paths=None):
    """
    Lets the user choose a raw dataset folder (or use the provided folder)
    and prepares it for YOLO training.

    Args:
        folder: Dataset root.  When None the user is asked to pick one.
        train_split: Fraction of pairs used for training.
        image_paths: Optional explicit list of images to use.  When given, only
            these are used instead of every image found under folder.

    Returns:
        Path to the prepared dataset, or None if it could not be prepared.
    """
    if folder is None:
        ensure_app()
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Raw Dataset Folder"
        )

    if not folder or not os.path.isdir(folder):
        show_error_window(
            "No valid dataset folder was selected.", title="Dataset Not Found"
        )
        return None

    # Output folder inside temp.  Clear it first: leftovers from an earlier run
    # would otherwise be folded into this training set.
    output_dir = os.path.join(tempfile.gettempdir(), "yolo_dataset")
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataset
    if (
        prepare_yolo_dataset(
            folder,
            output_dir,
            train_split=train_split,
            class_names=["defect"],
            image_paths=image_paths,
        )
        is None
    ):
        return None

    return output_dir  # path with YOLO-ready dataset + dataset.yaml


def prepare_yolo_dataset(
    input_dir, output_dir, train_split=0.8, class_names=None, image_paths=None
):
    """
    Prepares a YOLOv8-ready dataset structure by scanning all subdirectories for image+txt files.

    Args:
        input_dir: Dataset root to scan.
        output_dir: Where the YOLO-ready structure is written.
        train_split: Fraction of pairs used for training.
        class_names: Class names written into dataset.yaml.
        image_paths: Optional explicit list of images; when given, input_dir is
            not scanned and only these images (plus their .txt labels) are used.

    Returns:
        Path to the written dataset.yaml, or None on failure.
    """
    if class_names is None:
        class_names = ["defect"]

    # Create output dirs
    images_train = os.path.join(output_dir, "images/train")
    images_val = os.path.join(output_dir, "images/val")
    labels_train = os.path.join(output_dir, "labels/train")
    labels_val = os.path.join(output_dir, "labels/val")
    for d in [images_train, images_val, labels_train, labels_val]:
        os.makedirs(d, exist_ok=True)

    # Collect image+label pairs
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    data_pairs = []

    if image_paths is not None:
        candidates = [(os.path.dirname(p), os.path.basename(p)) for p in image_paths]
    else:
        candidates = [
            (root, f) for root, _dirs, files in os.walk(input_dir) for f in files
        ]

    for root, f in candidates:
        name, ext = os.path.splitext(f)
        if ext.lower() not in exts:
            continue
        img_path = os.path.join(root, f)
        label_path = os.path.join(root, name + ".txt")
        if os.path.isfile(img_path) and os.path.isfile(label_path):
            data_pairs.append((img_path, label_path))

    if not data_pairs:
        show_error_window(
            "No valid image-label pairs found in the selected dataset folder."
        )
        return None

    if len(data_pairs) < 2:
        show_error_window(
            "At least 2 labelled images are needed so that both a training and "
            "a validation set can be created.",
            title="Not Enough Images",
        )
        return None

    # Train/val split.  The count is clamped so neither side is ever empty,
    # which train_test_split would otherwise reject outright.
    n_train = max(1, min(len(data_pairs) - 1, round(train_split * len(data_pairs))))
    train_pairs, val_pairs = train_test_split(
        data_pairs, train_size=n_train, shuffle=True, random_state=42
    )

    # Copy files.  The structure is flat, so names coming from different
    # subdirectories are de-duplicated instead of silently overwriting.
    used_names = set()

    def copy_pairs(pairs, img_dest, label_dest):
        for img, lbl in pairs:
            name, ext = os.path.splitext(os.path.basename(img))
            unique = name
            suffix = 1
            while unique in used_names:
                unique = f"{name}_{suffix}"
                suffix += 1
            used_names.add(unique)
            shutil.copy(img, os.path.join(img_dest, unique + ext))
            shutil.copy(lbl, os.path.join(label_dest, unique + ".txt"))

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
