# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore


class InputDialogInference(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        font = QtGui.QFont()
        font.setPointSize(14)
        self.setFont(font)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Input Parameters")
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self.show_current_image)
        self.close_flag = False

        def make_label_with_tooltip(text, tooltip):
            container = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            label = QtWidgets.QLabel(text)
            icon_label = QtWidgets.QLabel()
            icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
            icon_label.setPixmap(icon_pix.pixmap(20, 20))
            icon_label.setToolTip(tooltip)
            layout.addWidget(label)
            layout.addWidget(icon_label)
            layout.addStretch()
            return container

        # === Controls ===
        self.image_path_edit = QtWidgets.QLineEdit()
        self.browse_image_button = QtWidgets.QPushButton("Browse Folder of Images...")

        self.output_path_edit = QtWidgets.QLineEdit()
        self.browse_output_button = QtWidgets.QPushButton("Browse Output Folder...")

        # === Model Selection ===
        self.model_path_edit = QtWidgets.QLineEdit()
        self.browse_model_button = QtWidgets.QPushButton("Select trained YOLO Model...")

        model_label = make_label_with_tooltip(
            "Trained YOLO Model:",
            "Select the trained YOLO model to use for object detection. This should be a model that you trained previously using the training workflow. This should be a .pt file.",
        )

        self.confidence_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)  # slider values 0–100
        self.confidence_slider.setValue(50)  # default 0.5
        self.confidence_slider.setSingleStep(1)

        self.confidence_spinbox = QtWidgets.QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setSingleStep(0.01)
        self.confidence_spinbox.setDecimals(2)
        self.confidence_spinbox.setValue(0.5)

        # Link slider <-> spinbox
        self.confidence_slider.valueChanged.connect(
            lambda val: self.confidence_spinbox.setValue(val / 100)
        )
        self.confidence_spinbox.valueChanged.connect(
            lambda val: self.confidence_slider.setValue(int(val * 100))
        )

        # Clamp spinbox input to 0–1
        self.confidence_spinbox.editingFinished.connect(
            lambda: self.confidence_spinbox.setValue(
                min(max(self.confidence_spinbox.value(), 0.0), 1.0)
            )
        )

        conf_container = QtWidgets.QWidget()
        outer_layout = QtWidgets.QVBoxLayout(conf_container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(4)

        # --- Top row: label + icon ---
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(4)

        conf_label = QtWidgets.QLabel("Confidence Threshold:")
        icon_label = QtWidgets.QLabel()
        icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
        icon_label.setPixmap(icon_pix.pixmap(20, 20))
        icon_label.setToolTip(
            "Set the confidence threshold for objects to be detected by the YOLO model. "
            "Higher values mean the model must be more certain before it labels an object."
        )

        top_row.addWidget(conf_label)
        top_row.addWidget(icon_label)
        top_row.addStretch()

        # --- Bottom row: slider + spinbox ---
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(4)

        bottom_row.addWidget(self.confidence_slider)
        bottom_row.addWidget(self.confidence_spinbox)

        # --- Assemble ---
        outer_layout.addLayout(top_row)
        outer_layout.addLayout(bottom_row)

        # Initially hide; show only if bootstrapping model is selected
        self.confidence_container = conf_container

        # === Run Button ===
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14pt;
            }
            QPushButton:hover { background-color: #005a9e; }
            QPushButton:pressed { background-color: #004578; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)

        self.status_label = QtWidgets.QLabel()
        palette = self.status_label.palette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("red"))
        self.status_label.setPalette(palette)
        self.status_label.setWordWrap(True)
        font = self.status_label.font()
        font.setPointSize(16)  # choose whatever size you want
        self.status_label.setFont(font)
        self.status_label.show()

        # === Image Preview and Navigation ===
        self.image_preview = QtWidgets.QLabel()
        self.image_preview.setStyleSheet(
            "border: 1px solid black; background-color: #eee;"
        )
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_preview.setStyleSheet("font-size: 16pt;")
        self.image_preview.setText("No Folder Selected")
        self.image_preview.setScaledContents(False)
        self.image_preview.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        # Keep track of folder images
        self.image_files = []
        self.current_image_index = -1

        # Navigation buttons
        self.prev_button = QtWidgets.QPushButton("◀ Previous")
        self.next_button = QtWidgets.QPushButton("Next ▶")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        preview_layout = QtWidgets.QVBoxLayout()
        preview_layout.addWidget(self.image_preview)
        preview_layout.addLayout(nav_layout)
        preview_widget = QtWidgets.QWidget()
        preview_widget.setLayout(preview_layout)

        # === Helper to create label + tooltip ===

        image_path_label = make_label_with_tooltip(
            "Image Folder:",
            "Select a folder containing input images to label for future training.",
        )
        output_path_label = make_label_with_tooltip(
            "Output Folder:",
            "Select a folder where any generated outputs will be saved.",
        )

        # === Controls Layout ===
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(image_path_label)
        controls_layout.addWidget(self.image_path_edit)
        controls_layout.addWidget(self.browse_image_button)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(output_path_label)
        controls_layout.addWidget(self.output_path_edit)
        controls_layout.addWidget(self.browse_output_button)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(model_label)
        controls_layout.addWidget(self.model_path_edit)
        controls_layout.addWidget(self.browse_model_button)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.confidence_container)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.run_button)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

        controls_widget = QtWidgets.QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setMinimumWidth(600)

        # === Main Layout ===
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(preview_widget, stretch=1)
        self.setLayout(main_layout)

        # === Connections ===
        self.browse_image_button.clicked.connect(self.browse_image)
        self.browse_model_button.clicked.connect(self.browse_model)
        self.browse_output_button.clicked.connect(self.browse_output)
        self.run_button.clicked.connect(self.on_run_clicked)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.image_path_edit.textChanged.connect(self.update_run_button_state)
        self.output_path_edit.textChanged.connect(self.update_run_button_state)
        self.model_path_edit.textChanged.connect(self.update_run_button_state)

        self.run_button.setEnabled(False)
        self.update_run_button_state()

        # Show dialog
        self.showMaximized()
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
        self.show()

    def browse_model(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "Model Files (*.pt *.pth *.onnx);;All Files (*)",
        )
        if file_path:
            self.model_path_edit.setText(file_path)

        if self.model_path_edit.text().strip():
            self.confidence_container.setVisible(True)
        else:
            self.confidence_container.setVisible(False)

    # === Folder Browse ===
    def browse_image(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder of Images"
        )
        if not folder:
            return

        self.image_path_edit.setText(folder)
        valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
        self.image_files = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(valid_exts)
        ]

        if not self.image_files:
            self.image_preview.setText("No image files found in folder.")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        self.current_image_index = 0
        self.show_current_image()
        self.update_navigation_buttons()

    # === Image Navigation ===
    def show_current_image(self):
        if not self.image_files:
            self.image_preview.setText("No images loaded.")
            return

        image_path = self.image_files[self.current_image_index]

        if getattr(self, "_original_pixmap_path", None) != image_path:
            self._original_pixmap = QtGui.QPixmap(image_path)
            self._original_pixmap_path = image_path
            self._last_scaled_size = None

        if self._original_pixmap.isNull():
            self.image_preview.setText("Failed to load image")
            return

        label_size = self.image_preview.size()
        w, h = label_size.width(), label_size.height()
        rounded_size = (w // 10 * 10, h // 10 * 10)

        if getattr(self, "_last_scaled_size", None) == rounded_size:
            return
        self._last_scaled_size = rounded_size

        scaled_pixmap = self._original_pixmap.scaled(
            label_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.image_preview.setPixmap(scaled_pixmap)

    def show_next_image(self):
        if not self.image_files:
            return
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_current_image()
        self.update_navigation_buttons()

    def show_previous_image(self):
        if not self.image_files:
            return
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(
            self.current_image_index < len(self.image_files) - 1
        )

    # === Output Folder ===
    def browse_output(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        )
        if folder:
            self.output_path_edit.setText(folder)

    # === Button Enable Logic ===
    def update_run_button_state(self):
        image_path = self.image_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()
        YOLO_model = self.model_path_edit.text().strip()

        errors = []
        if not image_path:
            errors.append("Please select an image folder.")
        elif not os.path.isdir(image_path):
            errors.append("Selected path must be a folder.")

        if not output_path:
            errors.append("Please select an output folder.")

        if not os.path.isfile(YOLO_model) or not YOLO_model.lower().endswith(
            (".pt", ".pth", ".onnx")
        ):
            errors.append("YOLO model path is invalid.")

        if errors:
            self.run_button.setEnabled(False)
            self.status_label.setText("\n".join(errors))
            self.status_label.show()
        else:
            self.run_button.setEnabled(True)
            self.status_label.hide()

    def on_run_clicked(self):
        """Called when the Run button is clicked"""
        self.close_flag = True  # window can close without quitting program
        self.accept()

    def closeEvent(self, event):
        """Called when the window is closed"""
        if getattr(self, "close_flag", False):
            # Run button triggered — just close dialog, do not exit program
            event.accept()
        else:
            # User clicked X — fully exit program
            print("Window closed — exiting program.")
            QtWidgets.QApplication.quit()
            sys.exit(0)

    # === Misc ===
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Escape:
            sys.exit(0)
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_current_image()

    def get_values(self):
        return {
            "image_paths": self.image_files,
            "output_folder": self.output_path_edit.text().strip(),
            "YOLO_model": self.model_path_edit.text().strip() or None,
            "YOLO_confidence": (
                self.confidence_spinbox.value()
                if self.model_path_edit.text().strip()
                else None
            ),
        }


def get_user_inference_inputs():
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if not app:
        app = QtWidgets.QApplication(sys.argv)
        owns_app = True

    dialog = InputDialogInference()
    result = None
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        result = dialog.get_values()

    if owns_app:
        app.quit()
    return result
