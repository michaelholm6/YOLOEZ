# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore

class InputDialog(QtWidgets.QDialog):
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

        self.suppress_checkbox = QtWidgets.QCheckBox()

        self.output_path_edit = QtWidgets.QLineEdit()
        self.browse_output_button = QtWidgets.QPushButton("Browse Output Folder...")
        
        
        # === Bootstrapping Model Selection ===
        self.model_path_edit = QtWidgets.QLineEdit()
        self.browse_model_button = QtWidgets.QPushButton("Select Bootstrapping Model...")

        model_label = make_label_with_tooltip(
            "Bootstrapping Model:",
            "Optional: Select a pre-trained bootstrapping model to use. This will automatically generate initial annotations for your images. \
You will have a chance to edit these annotations later. This should be a .pt file, and can be a model you have previously trained using this tool."
        )
        
        self.confidence_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)  # slider values 0–100
        self.confidence_slider.setValue(50)      # default 0.5
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
        conf_container.setVisible(False)
        self.confidence_container = conf_container


        # === Mode Selection (Bounding Box vs Segmentation) ===
        
        mode_label = make_label_with_tooltip(
            "Annotation Mode:",
            "Choose how you want to annotate images: Bounding Boxes or Segmentation Masks. Segmentations masks are many sided polygons that can better fit irregular shapes, while bounding boxes are simple rectangles."
        )
        mode_label.setToolTip("Choose how you want to annotate images: Bounding Boxes or Segmentation Masks.")

        self.bbox_radio = QtWidgets.QRadioButton("Bounding Boxes")
        self.segmentation_radio = QtWidgets.QRadioButton("Segmentation")

        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.addButton(self.bbox_radio)
        self.mode_group.addButton(self.segmentation_radio)

        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(self.bbox_radio)
        mode_layout.addWidget(self.segmentation_radio)
        mode_layout.addStretch()
        mode_container = QtWidgets.QWidget()
        mode_container.setLayout(mode_layout)

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
        font.setPointSize(16)   # choose whatever size you want
        self.status_label.setFont(font)
        self.status_label.show()

        # === Image Preview and Navigation ===
        self.image_preview = QtWidgets.QLabel()
        self.image_preview.setStyleSheet("border: 1px solid black; background-color: #eee;")
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_preview.setStyleSheet("font-size: 16pt;")
        self.image_preview.setText("No Folder Selected")
        self.image_preview.setScaledContents(False)
        self.image_preview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

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
            "Image Folder:", "Select a folder containing input images to label for future training."
        )
        output_path_label = make_label_with_tooltip(
            "Output Folder:", "Select a folder where any generated outputs will be saved."
        )

        # === YOLO Save Checkbox ===
        self.save_yolo_checkbox = QtWidgets.QCheckBox()
        yolo_container = QtWidgets.QWidget()
        yolo_layout = QtWidgets.QHBoxLayout(yolo_container)
        yolo_layout.setContentsMargins(0, 0, 0, 0)
        yolo_layout.setSpacing(4)
        yolo_layout.addWidget(self.save_yolo_checkbox)
        yolo_label = QtWidgets.QLabel("Save YOLO Training Sample")
        yolo_layout.addWidget(yolo_label)
        yolo_icon = QtWidgets.QLabel()
        icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
        yolo_icon.setPixmap(icon_pix.pixmap(20, 20))
        yolo_icon.setToolTip("Check to save a YOLO training samples for future model training based on your annotations.")
        yolo_layout.addWidget(yolo_icon)
        yolo_layout.addStretch()
        
        #Save unlabeled images checkbox
        self.save_unlabeled_checkbox = QtWidgets.QCheckBox()
        unlabeled_container = QtWidgets.QWidget()
        unlabeled_layout = QtWidgets.QHBoxLayout(unlabeled_container)
        unlabeled_layout.setContentsMargins(0, 0, 0, 0)
        unlabeled_layout.setSpacing(4)
        unlabeled_layout.addWidget(self.save_unlabeled_checkbox)
        unlabeled_label = QtWidgets.QLabel("Save Unlabeled Images")
        unlabeled_layout.addWidget(unlabeled_label)
        unlabeled_icon = QtWidgets.QLabel()
        icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
        unlabeled_icon.setPixmap(icon_pix.pixmap(20, 20))
        unlabeled_icon.setToolTip("Check to save images that do not contain any labeled objects to the YOLO training sample. This is off by default, because\
 the default behavior is to assume that unlabeled images are unlabeled because you ran out of time trying to label everything. However, it is important to train\
 your model with images that do not contain objects of interest, so you may want to enable this option.")
        unlabeled_layout.addWidget(unlabeled_icon)
        unlabeled_layout.addStretch()
        

        # === Controls Layout ===
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(image_path_label)
        controls_layout.addWidget(self.image_path_edit)
        controls_layout.addWidget(self.browse_image_button)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(mode_label)
        controls_layout.addWidget(mode_container)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(yolo_container)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(unlabeled_container)
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
        self.run_button.clicked.connect(self.accept)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.image_path_edit.textChanged.connect(self.update_run_button_state)
        self.output_path_edit.textChanged.connect(self.update_run_button_state)
        self.bbox_radio.toggled.connect(self.update_run_button_state)
        self.segmentation_radio.toggled.connect(self.update_run_button_state)

        self.run_button.setEnabled(False)
        self.update_run_button_state()

        # Show dialog
        
        self.showMaximized()
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
        self.show()
        
    def browse_model(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Bootstrapping Model",
            "",
            "Model Files (*.pt *.pth *.onnx);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            
        if self.model_path_edit.text().strip():
            self.confidence_container.setVisible(True)
        else:
            self.confidence_container.setVisible(False)
    

    # === Folder Browse ===
    def browse_image(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder of Images")
        if not folder:
            return

        self.image_path_edit.setText(folder)
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
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
        rounded_size = (w//10*10, h//10*10)

        if getattr(self, "_last_scaled_size", None) == rounded_size:
            return
        self._last_scaled_size = rounded_size

        scaled_pixmap = self._original_pixmap.scaled(
            label_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
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
        self.next_button.setEnabled(self.current_image_index < len(self.image_files) - 1)

    # === Output Folder ===
    def browse_output(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_path_edit.setText(folder)

    # === Button Enable Logic ===
    def update_run_button_state(self):
        image_path = self.image_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()
        mode_selected = self.bbox_radio.isChecked() or self.segmentation_radio.isChecked()
        bootstrapping_model = self.model_path_edit.text().strip()

        errors = []
        if not image_path:
            errors.append("Please select an image folder.")
        elif not os.path.isdir(image_path):
            errors.append("Selected path must be a folder.")

        if not output_path:
            errors.append("Please select an output folder.")

        if not mode_selected:
            errors.append("Please select either Bounding Boxes or Segmentation.")
            
        if bootstrapping_model:
            if not os.path.isfile(bootstrapping_model) or not bootstrapping_model.lower().endswith(('.pt', '.pth', '.onnx')):
                errors.append("Bootstrapping model path is invalid.")

        if errors:
            self.run_button.setEnabled(False)
            self.status_label.setText("\n".join(errors))
            self.status_label.show()
        else:
            self.run_button.setEnabled(True)
            self.status_label.hide()

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
        mode = "bounding_box" if self.bbox_radio.isChecked() else "segmentation"
        return {
            "image_paths": self.image_files,
            "suppress_instructions": self.suppress_checkbox.isChecked(),
            "output_folder": self.output_path_edit.text().strip(),
            "YOLO_true": self.save_yolo_checkbox.isChecked(),
            "annotation_mode": mode,
            "bootstrapping_model": self.model_path_edit.text().strip() or None,
            "bootstrapping_confidence": self.confidence_spinbox.value() if self.model_path_edit.text().strip() else None,
            "save_unlabeled_images": self.save_unlabeled_checkbox.isChecked()
        }


def get_user_labelling_inputs():
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
