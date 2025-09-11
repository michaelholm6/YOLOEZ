from PyQt5 import QtWidgets
import sys, shutil

def get_save_path(default="model.pth"):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        None, "Save Trained Model", default, "PyTorch Model (*.pth)"
    )
    return path

def save_trained_model(source_path, dest_path):
    if dest_path:
        shutil.copy(source_path, dest_path)
        return dest_path
    return None

if __name__ == "__main__":
    # Example: copy YOLOv8's best.pt
    model_out = get_save_path()
    if model_out:
        save_trained_model("runs/detect/train/weights/best.pt", model_out)
        print("Model saved at:", model_out)