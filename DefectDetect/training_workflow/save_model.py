from PyQt5 import QtWidgets
import sys, shutil, os

def get_save_root(default_dir=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    root = QtWidgets.QFileDialog.getExistingDirectory(
        None,
        "Select Folder to Save Trained Model",
        default_dir or ""
    )
    return root

def save_trained_model(source_path, dest_root, filename="best.pt"):
    if not dest_root:
        return None

    os.makedirs(dest_root, exist_ok=True)
    dest_path = os.path.join(dest_root, filename)

    shutil.copy(source_path, dest_path)
    return dest_path

if __name__ == "__main__":
    # Example: copy YOLO best.pt
    save_root = get_save_root()

    if save_root:
        saved_path = save_trained_model(
            source_path="runs/detect/train/weights/best.pt",
            dest_root=save_root,
            filename="best.pt"
        )
        print("✅ Model saved at:", saved_path)
