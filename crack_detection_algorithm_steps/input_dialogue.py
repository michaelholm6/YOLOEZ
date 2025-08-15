# input_dialog.py (updated size)
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np


class InputDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle("Input Parameters")

        # === Controls ===
        self.image_path_edit = QtWidgets.QLineEdit()
        self.browse_image_button = QtWidgets.QPushButton("Browse Image...")

        self.suppress_checkbox = QtWidgets.QCheckBox()

        self.output_path_edit = QtWidgets.QLineEdit()
        self.browse_output_button = QtWidgets.QPushButton("Browse Output Folder...")
        
        self.output_filename_edit = QtWidgets.QLineEdit()
        self.output_filename_edit.setPlaceholderText("e.g. result.png")

        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setStyleSheet("""
    QPushButton {
        background-color: #0078d7;
        color: white;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #005a9e;
    }
    QPushButton:pressed {
        background-color: #004578;
    }
    QPushButton:disabled {
        background-color: #cccccc;  /* light gray */
        color: #666666;            /* darker text */
    }
""")
        
        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet("color: red;")
        self.status_label.setWordWrap(True)
        self.status_label.show()

        # === Image Preview ===
        self.image_preview = QtWidgets.QLabel()
        self.image_preview.setStyleSheet("border: 1px solid black; background-color: #eee;")
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_preview.setText("No Image Selected")
        self.image_preview.setScaledContents(False)
        self.image_preview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Helper to create label + question mark icon with tooltip
        def make_label_with_tooltip(text, tooltip):
            container = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

            label = QtWidgets.QLabel(text)
            label.setToolTip(tooltip)

            # Small question mark icon
            icon_label = QtWidgets.QLabel()
            icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
            icon_label.setPixmap(icon_pix.pixmap(14, 14))
            icon_label.setToolTip(tooltip)

            layout.addWidget(label)
            layout.addWidget(icon_label)
            layout.addStretch()

            return container

        # Create labeled widgets with tooltips
        image_path_label = make_label_with_tooltip(
            "Image Path:", "Enter the path to the input image file here."
        )
        output_path_label = make_label_with_tooltip(
            "Output Path:", "Enter the output folder path where results will be saved."
        )
        output_filename_label = make_label_with_tooltip(
            "Output Filename:",
            "Specify the output file name including extension (e.g., 'result.png', 'output.jpg')."
        )

        # Create checkbox + question mark for suppress instructions
        suppress_container = QtWidgets.QWidget()
        suppress_layout = QtWidgets.QHBoxLayout(suppress_container)
        suppress_layout.setContentsMargins(0, 0, 0, 0)
        suppress_layout.setSpacing(4)
        suppress_layout.addWidget(self.suppress_checkbox)

        suppress_label = QtWidgets.QLabel("Suppress Instructions")
        suppress_label.setToolTip("Check this to suppress instructions during processing.")
        suppress_layout.addWidget(suppress_label)

        suppress_icon = QtWidgets.QLabel()
        icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
        suppress_icon.setPixmap(icon_pix.pixmap(14, 14))
        suppress_icon.setToolTip("Check this to suppress instructions during processing.")
        suppress_layout.addWidget(suppress_icon)

        suppress_layout.addStretch()

        # === Controls Layout ===
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(image_path_label)
        controls_layout.addWidget(self.image_path_edit)
        controls_layout.addWidget(self.browse_image_button)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(suppress_container)   # just one line here now
        controls_layout.addSpacing(10)
        controls_layout.addWidget(output_path_label)
        controls_layout.addWidget(self.output_path_edit)
        controls_layout.addWidget(self.browse_output_button)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(output_filename_label)
        controls_layout.addWidget(self.output_filename_edit)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.run_button)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

        # === Controls container with wider default width ===
        controls_widget = QtWidgets.QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setMinimumWidth(600)

        # === Main Layout ===
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(self.image_preview, stretch=1)

        self.setLayout(main_layout)

        # === Connections ===
        self.browse_image_button.clicked.connect(self.browse_image)
        self.browse_output_button.clicked.connect(self.browse_output)
        self.run_button.clicked.connect(self.accept)
        
        self.image_path_edit.textChanged.connect(self.update_run_button_state)
        self.output_path_edit.textChanged.connect(self.update_run_button_state)
        self.output_filename_edit.textChanged.connect(self.update_run_button_state)

        # Disable button initially
        self.run_button.setEnabled(False)
        self.update_run_button_state()

        # Important: call showMaximized() AFTER setting layout
        self.showMaximized()
        
    def update_run_button_state(self):
        image_path = self.image_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()
        filename = self.output_filename_edit.text().strip()

        valid_exts = ('.jpg', '.png', '.tiff')

        errors = []

        if not image_path:
            errors.append("Please select an input image.")
        elif not image_path.lower().endswith(valid_exts):
            errors.append("Input image must end with .jpg, .png, or .tiff.")

        if not output_path:
            errors.append("Please select an output folder.")

        if not filename:
            errors.append("Please enter an output file name.")
        elif not filename.lower().endswith(valid_exts):
            errors.append("Output file name must end with .jpg, .png,  .tif or .tiff.")

        if errors:
            self.run_button.setEnabled(False)
            self.status_label.setText("\n".join(errors))
            self.status_label.show()
        else:
            self.run_button.setEnabled(True)
            self.status_label.hide() 
    
    def keyPressEvent(self, event: QtGui.QKeyEvent):
            if event.key() == QtCore.Qt.Key_Escape:
                sys.exit(0)  # quit python script entirely
            else:
                super().keyPressEvent(event)

    def browse_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.image_path_edit.setText(path)

            pixmap = QtGui.QPixmap(path)

            # Get available preview area
            preview_size = self.image_preview.size()

            if preview_size.width() <= 0 or preview_size.height() <= 0:
                # fallback guess
                preview_size = QtCore.QSize(800, 600)

            # Scale pixmap to fit preview area while keeping aspect ratio
            scaled_pixmap = pixmap.scaled(
                preview_size,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.image_preview.setPixmap(scaled_pixmap)
            self.image_preview.setText("")  # clear text

    def browse_output(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_path_edit.setText(folder)

    def show_image_preview(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            self.image_preview.setText("Failed to load image")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_img = img
        self.update_image_preview_size()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_preview_size()

    def update_image_preview_size(self):
        if not hasattr(self, "current_img"):
            return

        img = self.current_img.copy()
        h_img, w_img, _ = img.shape

        # Available area for image
        total_w = self.width()
        total_h = self.height()

        # Estimate controls height
        controls_height = 320  # Adjust if your controls are taller

        max_w = total_w - 100  # Margins
        max_h = total_h - controls_height - 50

        # Scale UP and DOWN as needed to fit available space
        scale = min(max_w / w_img, max_h / h_img)

        new_w, new_h = int(w_img * scale), int(h_img * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        qimg = QtGui.QImage(img_resized.data, new_w, new_h, 3 * new_w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)

        self.image_preview.setPixmap(pixmap)
        self.image_preview.setFixedSize(new_w, new_h)

    def get_values(self):
        folder = self.output_path_edit.text().strip()
        filename = self.output_filename_edit.text().strip()

        # Join folder + filename if both are given
        if folder and filename:
            import os
            full_output_path = os.path.join(folder, filename)
        else:
            full_output_path = filename or folder  # whichever is given or empty

        return {
            "image_path": self.image_path_edit.text(),
            "suppress_instructions": self.suppress_checkbox.isChecked(),
            "output_path": full_output_path
        }

def get_user_inputs():
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if not app:
        app = QtWidgets.QApplication(sys.argv)
        owns_app = True

    dialog = InputDialog()
    result = None
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        result = dialog.get_values()

    if owns_app:
        app.quit()
    return result