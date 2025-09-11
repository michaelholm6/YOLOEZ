# dataset_chooser.py
from PyQt5 import QtWidgets
import sys
import os
import shutil
import random
from sklearn.model_selection import train_test_split

def choose_dataset(train_split=0.8):
    """
    Lets the user choose a raw dataset folder and prepares it for YOLO training.
    Walks all subdirectories and collects training pairs.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    folder = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Raw Dataset Folder")
    if folder and os.path.isdir(folder):
        # Output folder inside the selected folder
        output_dir = os.path.join(folder, "yolo_dataset")
        os.makedirs(output_dir, exist_ok=True)

        # Always use single class
        prepare_yolo_dataset(folder, output_dir, train_split=train_split, class_names=["defect"])

        return output_dir  # path with YOLO-ready dataset + dataset.yaml
    else:
        return None


def prepare_yolo_dataset(input_dir, output_dir, train_split=0.8, class_names=None):
    """
    Prepares a YOLOv8-ready dataset structure by scanning all subdirectories for image+txt files.

    Args:
        input_dir (str): Root folder containing subfolders with images and labels
        output_dir (str): Output folder where YOLO structure will be created
        train_split (float): Fraction of data to use for training
        class_names (list[str]): List of class names. Defaults to ["defect"]
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

    # Collect pairs (image, label) recursively
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
        raise RuntimeError(f"No image+label pairs found in {input_dir}")

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


if __name__ == "__main__":
    path = choose_dataset(train_split=0.8)
    print("Prepared YOLO dataset at:", path)