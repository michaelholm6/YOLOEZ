from PyQt5 import QtWidgets, QtGui, QtCore
import sys

class AugmentationDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        font = QtGui.QFont()
        font.setPointSize(14)  # increase this value as needed
        self.setFont(font)
        self.setWindowTitle("Augmentation Options")
        self.setMinimumWidth(300)

        layout = QtWidgets.QVBoxLayout()

        self.options = [
            ("Flip", "Add horizontally flipped images to the training set"),
            ("Rotate", "Add rotated images to the training set"),
            ("Color Jitter", "Add color jittering to the dataset. Randomly changes brightness, contrast, saturation, and hue"),
            ("Blur", "Add blurred images to the training set"),
            ("Noise", "Add images with random noise to the training set"),
            ("Scale", "Add scaled images to the training set"),
        ]

        self.checkboxes = []
        for text, tooltip in self.options:
            container = self.make_label_with_tooltip(text, tooltip)
            cb = QtWidgets.QCheckBox()
            self.checkboxes.append(cb)

            row = QtWidgets.QHBoxLayout()
            row.addWidget(cb)
            row.addWidget(container)
            row.addStretch()
            layout.addLayout(row)

        # --- New field for number of augmentations ---
        aug_count_layout = QtWidgets.QHBoxLayout()
        screen = QtWidgets.QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch()
        aug_label = QtWidgets.QLabel("Augmentations per image:")
        aug_spin = QtWidgets.QSpinBox()
        aug_spin.setMinimum(1)
        aug_spin.setMaximum(100)
        aug_spin.setValue(1)  # default
        self.aug_count_spin = aug_spin  # save for get_aug_dict()

        aug_count_layout.addWidget(aug_label)
        aug_count_layout.addWidget(aug_spin)
        aug_count_layout.addStretch()
        layout.addLayout(aug_count_layout)
        # --- End new field ---

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.show()
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
        self.show()

    def make_label_with_tooltip(self, text, tooltip_text):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QtWidgets.QLabel(text)
        label.setToolTip(tooltip_text)

        icon_label = QtWidgets.QLabel()
        icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
        icon_label.setPixmap(icon_pix.pixmap(16, 16))
        icon_label.setToolTip(tooltip_text)

        layout.addWidget(label)
        layout.addWidget(icon_label)
        return container

    def get_aug_dict(self):
        """Return dictionary compatible with build_transform(), plus augmentation count."""
        selected = [self.options[i][0] for i, cb in enumerate(self.checkboxes) if cb.isChecked()]
        return {
            "flip": "Flip" in selected,
            "rotate": "Rotate" in selected,
            "scale": "Scale" in selected,
            "color": "Color Jitter" in selected,
            "blur": "Blur" in selected,
            "noise": "Noise" in selected,
            "num_aug_per_image": self.aug_count_spin.value()  # new field
        }

        
def get_augmentations():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dialog = AugmentationDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        return dialog.get_aug_dict(), dialog.aug_count_spin.value()
    return {
        "flip": False,
        "rotate": False,
        "scale": False,
        "color": False,
        "blur": False,
        "noise": False
    }, 0