from PyQt5 import QtWidgets
import sys

def choose_images():
    """
    Opens a file dialog to let the user select one or more image files for inference.

    Returns:
        List of selected file paths (strings). Returns empty list if no selection.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    
    # Open file dialog
    file_dialog = QtWidgets.QFileDialog()
    file_dialog.setWindowTitle("Select Images for Inference")
    file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)  # multiple selection allowed
    file_dialog.setNameFilters(["Images (*.png *.jpg *.jpeg *.bmp *.tiff)"])
    
    if file_dialog.exec_():
        selected_files = file_dialog.selectedFiles()
        return selected_files
    
    return []