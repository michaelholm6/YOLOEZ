# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import sys, os
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np

class YOLOTrainingDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Training Setup")
        font = QtGui.QFont()
        font.setPointSize(14)
        self.setFont(font)
        self.task = "segmentation"
        self.close_flag = False

        # --- Helper function for tooltips ---
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

        # --- Left pane: controls ---
        left_layout = QtWidgets.QVBoxLayout()

        # Dataset selection
        dataset_container = make_label_with_tooltip(
            "Training Dataset Folder:",
            "Select the root folder containing your training images and YOLO label txt files created using this tool."
        )

        self.dataset_path_edit = QtWidgets.QLineEdit()
        self.browse_dataset_button = QtWidgets.QPushButton("Browse Training Dataset...")
        left_layout.addWidget(dataset_container)
        left_layout.addWidget(self.dataset_path_edit)
        left_layout.addWidget(self.browse_dataset_button)

        # Transformations
        # --- Transformations ---
        apply_transforms_label = make_label_with_tooltip(
            "Apply Data Augmentations:", "Data augmentations help improve model robustness by artificially increasing dataset diversity. Select the augmentations you want to apply during training.\
Baiscally this will create more images from your existing ones by applying these transformations, and use those images for training.")
        left_layout.addWidget(apply_transforms_label)
        self.transform_options = [
            ("Flip", "Randomly horizontally flip some images."),
            ("Rotate", "Randomly rotate some images. Disabled for detection tasks."),
            ("Color Jitter", "Randomly adjust brightness/contrast/saturation/hue of some images."),
            ("Blur", "Randomly apply Gaussian blur to some photos."),
            ("Noise", "Randomly add Gaussian noise to some photos."),
            ("Scale", "Randomly scale some photos."),
        ]
        self.transform_checkboxes = []

        for text, tooltip in self.transform_options:
            container = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

            cb = QtWidgets.QCheckBox(text)
            cb.setToolTip(tooltip)
            layout.addWidget(cb)

            icon_label = QtWidgets.QLabel()
            icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
            icon_label.setPixmap(icon_pix.pixmap(20, 20))
            icon_label.setToolTip(tooltip)
            layout.addWidget(icon_label)

            layout.addStretch()
            container.setLayout(layout)
            left_layout.addWidget(container)
            self.transform_checkboxes.append(cb)

            # Keep reference to the container for showing/hiding
            if text == "Rotate":
                self.rotate_container = container

        # --- Hide Rotate if detection ---
        if self.task == "detection":
            self.rotate_container.hide()
        else:
            self.rotate_container.show()


        # Number of augmentations per image
        aug_layout = QtWidgets.QHBoxLayout()
        aug_label_container = make_label_with_tooltip(
            "Augmentations per image:",
            "Set how many augmented versions of each image to generate."
        )
        self.aug_spin = QtWidgets.QSpinBox()
        self.aug_spin.setMinimum(1)
        self.aug_spin.setMaximum(100)
        self.aug_spin.setValue(1)
        aug_layout.addWidget(aug_label_container)
        aug_layout.addWidget(self.aug_spin)
        left_layout.addLayout(aug_layout)

        # YOLO model size
        model_label_container = make_label_with_tooltip(
            "YOLO Model Size:",
            "Select the model size (nano, small, medium, large, extra-large) to train. Larger models are typically more accurate but require more computational resources. Choosing a model that is larger\
than what your hardware can handle can lead to unexplained crashes during training."
        )
        left_layout.addWidget(model_label_container)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["Nano", "Small", "Medium", "Large", "Extra-Large"])
        left_layout.addWidget(self.model_combo)
        
        split_container = make_label_with_tooltip(
            "Train/Test Split (% for training):",
            "Specify the fraction of images used for training. The rest will be used for validation. Validation is a set that's not seen during training and is used to evaluate model performance throughout the training process.\
A typical split is 80% for training and 20% for validation."
        )
        self.split_spin = QtWidgets.QSpinBox()
        self.split_spin.setMinimum(1)
        self.split_spin.setMaximum(99)
        self.split_spin.setValue(80)  # default 80%
        split_layout = QtWidgets.QHBoxLayout()
        split_layout.addWidget(split_container)
        split_layout.addWidget(self.split_spin)
        left_layout.addLayout(split_layout)

        # --- Previously Trained Model ---
        prev_model_container = make_label_with_tooltip(
            "Continue Training from Model:",
            "Optional: Select a previously trained YOLO model to continue training. This should be a .pt file."
        )
        self.prev_model_edit = QtWidgets.QLineEdit()
        self.browse_prev_model_button = QtWidgets.QPushButton("Browse Model...")
        left_layout.addWidget(prev_model_container)
        prev_model_layout = QtWidgets.QHBoxLayout()
        prev_model_layout.addWidget(self.prev_model_edit)
        prev_model_layout.addWidget(self.browse_prev_model_button)
        left_layout.addLayout(prev_model_layout)

        # --- Connection ---
        self.browse_prev_model_button.clicked.connect(self.browse_prev_model)

        # Save directory
        save_container = make_label_with_tooltip(
            "Save Directory:",
            "Select the folder where the trained model and results will be saved."
        )
        self.save_path_edit = QtWidgets.QLineEdit()
        self.browse_save_button = QtWidgets.QPushButton("Browse Save Directory...")
        left_layout.addWidget(save_container)
        left_layout.addWidget(self.save_path_edit)
        left_layout.addWidget(self.browse_save_button)
        self.dataset_path_edit.textChanged.connect(self.update_run_button_state)
        self.save_path_edit.textChanged.connect(self.update_run_button_state)
        self.browse_dataset_button.clicked.connect(lambda: self.update_run_button_state())
        self.browse_save_button.clicked.connect(lambda: self.update_run_button_state())

        # Run button
        run_container = QtWidgets.QWidget()
        run_layout = QtWidgets.QVBoxLayout(run_container)
        run_layout.setContentsMargins(0,0,0,0)
        run_layout.setSpacing(4)

        # Run button
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
        self.run_button.setEnabled(False)
        run_layout.addWidget(self.run_button)

        # Status label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: red;")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        font = self.status_label.font()
        font.setPointSize(16)   # choose whatever size you want
        self.status_label.setFont(font)
        run_layout.addWidget(self.status_label)

        # Add the container to your left_layout
        left_layout.addWidget(run_container)
        left_layout.addStretch()
        

        left_container = QtWidgets.QWidget()
        left_container.setLayout(left_layout)
        left_container.setMinimumWidth(400)

        # --- Right pane: image preview ---
        right_layout = QtWidgets.QVBoxLayout()
        self.image_preview = QtWidgets.QLabel("No Dataset Selected")
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_preview.setStyleSheet("border: 1px solid black; background-color: #eee; font-size: 16pt;")
        self.image_preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image_preview.setScaledContents(False)
        right_layout.addWidget(self.image_preview)

        nav_layout = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("◀ Previous")
        self.next_button = QtWidgets.QPushButton("Next ▶")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        right_layout.addLayout(nav_layout)

        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_layout)

        # --- Main layout ---
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(left_container)
        main_layout.addWidget(right_container, stretch=1)
        self.setLayout(main_layout)

        # --- Internal state ---
        self.image_files = []
        self.current_image_index = -1
        self.task = "segmentation"  # default
        self._original_pixmap = None

        # --- Connections ---
        self.browse_dataset_button.clicked.connect(self.browse_dataset)
        self.browse_save_button.clicked.connect(self.browse_save)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.run_button.clicked.connect(self.on_run_clicked)
        for cb in self.transform_checkboxes:
            cb.stateChanged.connect(self.disable_rotation_if_detection)
            
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.showMaximized()
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
        self.show()
        self.update_run_button_state()
        
    def on_run_clicked(self):
        """Called when the Run button is clicked"""
        self.close_flag = True  # window can close without quitting program
        self.accept()           # close the dialog
        
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
            
    def update_run_button_state(self):
        errors = []
        dataset_folder = self.dataset_path_edit.text().strip()
        save_folder = self.save_path_edit.text().strip()

        if not dataset_folder:
            errors.append("Please select a dataset folder.")
        elif not os.path.isdir(dataset_folder):
            errors.append("Dataset folder does not exist.")

        if not save_folder:
            errors.append("Please select a save directory.")
        elif not os.path.isdir(save_folder):
            errors.append("Save folder does not exist.")

        if errors:
            self.run_button.setEnabled(False)
            self.status_label.setText("\n".join(errors))
        else:
            self.run_button.setEnabled(True)
            self.status_label.setText("")

                
    def browse_prev_model(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Previously Trained Model",
            "",
            "Model Files (*.pt *.pth *.onnx);;All Files (*)"
        )
        if file_path:
            self.prev_model_edit.setText(file_path)
            
    def update_transform_visibility(self):
        if self.task == "detection":
            self.rotate_container.hide()
        else:
            self.rotate_container.show()
            for cb in self.transform_checkboxes:
                if cb.text() == "Rotate":
                    cb.setEnabled(True)

    def showEvent(self, event):
        """Ensure the dialog actually starts maximized."""
        super().showEvent(event)
        self.showMaximized()
        
    def detect_label_format(self, label_path):
        """Return 'detection', 'segmentation', or None for empty/unreadable labels."""
        with open(label_path, "r") as f:
            lines = [line.strip().split() for line in f if line.strip()]

        if not lines:
            return "empty"  # treat empty label as valid, but not for task locking
        elif all(len(l) == 5 for l in lines):
            return "detection"
        else:
            return "segmentation"

    # --- Browse functions ---
    def browse_dataset(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Training Dataset")
        if not folder:
            return

        self.dataset_path_edit.setText(folder)

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        label_exts = ('.txt',)

        self.image_files = []
        detected_task = None  # lock dataset type from first **non-empty** label

        for root, dirs, files in os.walk(folder):
            label_basenames = {
                os.path.splitext(f)[0]
                for f in files
                if f.lower().endswith(label_exts)
            }

            for f in sorted(files):
                if not f.lower().endswith(valid_exts):
                    continue

                base = os.path.splitext(f)[0]
                if base not in label_basenames:
                    continue

                img_path = os.path.join(root, f)
                label_path = os.path.join(root, base + ".txt")

                try:
                    label_type = self.detect_label_format(label_path)
                except Exception:
                    continue  # unreadable label → skip

                if label_type == "empty":
                    self.image_files.append(img_path)
                    continue

                # First non-empty label → lock dataset type
                if detected_task is None:
                    detected_task = label_type
                    self.task = detected_task

                # Only keep matching formats
                if label_type == detected_task:
                    self.image_files.append(img_path)
    
        if detected_task is None:
            self.image_files = []
            self.image_preview.setText("No valid images found.")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.current_image_index = -1
            return

        # Update UI
        self.update_transform_visibility()
        self.disable_rotation_if_detection()

        self.current_image_index = 0
        self.show_current_image()
        self.update_navigation_buttons()



    def browse_save(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if folder:
            self.save_path_edit.setText(folder)

    # --- Image navigation ---
    def show_current_image(self):
        if not self.image_files or self.current_image_index < 0:
            self.image_preview.setText("No images loaded.")
            return

        img_path = self.image_files[self.current_image_index]
        lbl_path = os.path.splitext(img_path)[0] + ".txt"

        img = cv2.imread(img_path)
        if img is None:
            self.image_preview.setText("Failed to load image")
            return
        h, w = img.shape[:2]

        # Draw YOLO labels
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) == 4:  # bbox
                        xc, yc, bw, bh = coords
                        x1 = int((xc - bw/2) * w)
                        y1 = int((yc - bh/2) * h)
                        x2 = int((xc + bw/2) * w)
                        y2 = int((yc + bh/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        pts = [(int(coords[i]*w), int(coords[i+1]*h)) for i in range(0, len(coords), 2)]
                        cv2.polylines(img, [np.array(pts)], isClosed=True, color=(0, 0, 255), thickness=2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qt_img = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0],
                              img_rgb.strides[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        label_size = self.image_preview.size()
        scaled_pixmap = pixmap.scaled(label_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_preview.setPixmap(scaled_pixmap)

    def show_next_image(self):
        if self.current_image_index < len(self.image_files)-1:
            self.current_image_index += 1
            self.show_current_image()
        self.update_navigation_buttons()

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.image_files)-1)

    # --- Transformations ---
    def get_transform_dict(self):
        selected = [cb.text() for cb in self.transform_checkboxes if cb.isChecked()]
        rotate_enabled = "Rotate" in selected
        if self.task == "detection":
            rotate_enabled = False
        return {
            "flip": "Flip" in selected,
            "rotate": rotate_enabled,
            "scale": "Scale" in selected,
            "color": "Color Jitter" in selected,
            "blur": "Blur" in selected,
            "noise": "Noise" in selected,
        }

    def disable_rotation_if_detection(self):
        for cb in self.transform_checkboxes:
            if cb.text() == "Rotate":
                if self.task == "detection":
                    cb.setChecked(False)
                    cb.setEnabled(False)
                else:
                    cb.setEnabled(True)

    # --- Get final values ---
    def get_values(self):
        model_size_map = {
            "Nano": "n", 
            "Small": "s",
            "Medium": "m",
            "Large": "l",
            "Extra-Large": "x"
        }
        
        return {
            "dataset_folder": self.dataset_path_edit.text().strip(),
            "save_folder": self.save_path_edit.text().strip(),
            "model_size": model_size_map[self.model_combo.currentText()],
            "task": self.task,
            "transformations": self.get_transform_dict(),
            "train_split": self.split_spin.value() / 100.0,
            "prev_model_path": self.prev_model_edit.text().strip() or None,
            "number_of_augs": self.aug_spin.value()
        }

def get_training_inputs():
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if not app:
        app = QtWidgets.QApplication(sys.argv)
        owns_app = True

    dialog = YOLOTrainingDialog()
    result = None
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        result = dialog.get_values()

    if owns_app:
        app.quit()
    return result

if __name__ == "__main__":
    inputs = get_training_inputs()
    print(inputs)
