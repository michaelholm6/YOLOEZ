from DefectDetect.labelling_workflow.image_preprocessing import *
from DefectDetect.labelling_workflow.area_of_interest_marking import *
from DefectDetect.labelling_workflow.detect_cracks import *
from DefectDetect.labelling_workflow.clip_cracks_to_area_of_interest import *
from DefectDetect.labelling_workflow.save_and_display_results import *
from DefectDetect.labelling_workflow.edit_contour_points import *
from DefectDetect.labelling_workflow.input_dialogue import *
from DefectDetect.labelling_workflow.main_labelling import run_labeling_workflow
from DefectDetect.training_workflow.main_training import run_training_workflow
from DefectDetect.inference_workflow.main_inference import run_inference_workflow
from PyQt5 import QtWidgets
import sys

class ChoiceDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        font = QtGui.QFont()
        font.setPointSize(14)
        self.setFont(font)
        self.setWindowTitle("Choose Mode")

        layout = QtWidgets.QVBoxLayout(self)

        # Main label
        label = QtWidgets.QLabel("What would you like to do?")
        layout.addWidget(label)

        # Buttons in a row
        button_layout = QtWidgets.QHBoxLayout()
        self.label_button = QtWidgets.QPushButton("Label Data")
        self.train_button = QtWidgets.QPushButton("Train YOLO Model")
        self.use_button = QtWidgets.QPushButton("Use Trained Model")

        button_layout.addWidget(self.label_button)
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.use_button)
        layout.addLayout(button_layout)

        # Checkbox below buttons
        self.checkbox = QtWidgets.QCheckBox("Suppress instructions")
        self.checkbox.setChecked(False)  # default: instructions are shown
        layout.addWidget(self.checkbox)

        # Track chosen button
        self.chosen = None
        self.label_button.clicked.connect(lambda: self.set_choice("label"))
        self.train_button.clicked.connect(lambda: self.set_choice("train"))
        self.use_button.clicked.connect(lambda: self.set_choice("use"))

    def set_choice(self, choice):
        self.chosen = choice
        self.accept()  # close dialog


def main():
    app = QtWidgets.QApplication(sys.argv)

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