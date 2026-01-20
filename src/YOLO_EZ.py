# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

from labelling_workflow.image_gatherer import *
from labelling_workflow.area_of_interest_marking import *
from labelling_workflow.clip_cracks_to_area_of_interest import *
from labelling_workflow.save_segmentation_results import *
from labelling_workflow.edit_contour_points import *
from labelling_workflow.get_user_inputs_labelling import *
from labelling_workflow.main_labelling import run_labeling_workflow
from training_workflow.main_training import run_training_workflow
from inference_workflow.main_inference import run_inference_workflow
from utils import make_label_with_tooltip
from PyQt5 import QtWidgets
import sys

class ChoiceDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Choose Mode")

        font = QtGui.QFont()
        font.setPointSize(14)
        self.setFont(font)

        main_layout = QtWidgets.QVBoxLayout(self)

        # Main label
        label = QtWidgets.QLabel("What would you like to do?")
        main_layout.addWidget(label)

        # Button row
        button_layout = QtWidgets.QHBoxLayout()

        def make_button_with_icon(text, tooltip):
            container = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            button = QtWidgets.QPushButton(text)
            icon = QtWidgets.QLabel()
            pix = QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.SP_MessageBoxQuestion
            ).pixmap(20, 20)
            icon.setPixmap(pix)
            icon.setToolTip(tooltip)

            row.addWidget(button)
            row.addWidget(icon)
            row.addStretch()

            return container, button

        label_container, self.label_button = make_button_with_icon(
            "Label Images",
            "Start the labeling workflow to label objects in your images. These labels can then be used to train a model."
        )

        train_container, self.train_button = make_button_with_icon(
            "Train Model",
            "Start the training workflow to train an object detection model using images labeled with the labeling workflow."
        )

        use_container, self.use_button = make_button_with_icon(
            "Use Model",
            "Start the inference workflow to use a trained model for object detection."
        )

        button_layout.addWidget(label_container)
        button_layout.addWidget(train_container)
        button_layout.addWidget(use_container)

        main_layout.addLayout(button_layout)

        # Checkbox row
        h_layout = QtWidgets.QHBoxLayout()
        self.checkbox = QtWidgets.QCheckBox()
        self.checkbox_label = make_label_with_tooltip(
            "Suppress Instructional Popups",
            "Check this box to skip instructional pop-ups in the workflows."
        )

        h_layout.addWidget(self.checkbox)
        h_layout.addWidget(self.checkbox_label)
        h_layout.addStretch()

        main_layout.addLayout(h_layout)

        # Signals
        self.label_button.clicked.connect(lambda: self.set_choice("label"))
        self.train_button.clicked.connect(lambda: self.set_choice("train"))
        self.use_button.clicked.connect(lambda: self.set_choice("use"))

        self.chosen = None

        self.show()
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def set_choice(self, choice):
        self.chosen = choice
        self.accept()  # close dialog


def main():
    app = QtWidgets.QApplication(sys.argv)

    # --- Check screen resolution ---
    screen = app.primaryScreen()
    size = screen.size()
    width, height = size.width(), size.height()

    recommended_width, recommended_height = 1920, 1080

    if width < recommended_width or height < recommended_height:
        QtWidgets.QMessageBox.warning(
            None,
            "Screen Resolution Warning",
            f"Your current screen resolution is {width}×{height}, which is smaller than "
            f"the recommended {recommended_width}×{recommended_height}.\n\n"
            "The application may not fit fully on the screen and some elements may appear clipped."
        )

    dialog = ChoiceDialog()
    dialog.exec_()

    suppress_instructions = dialog.checkbox.isChecked()

    if dialog.chosen == "label":
        run_labeling_workflow(suppress_instructions=suppress_instructions)
    elif dialog.chosen == "train":
        run_training_workflow(suppress_instructions=suppress_instructions)
    elif dialog.chosen == "use":
        run_inference_workflow(suppress_instructions=suppress_instructions)

    sys.exit()
