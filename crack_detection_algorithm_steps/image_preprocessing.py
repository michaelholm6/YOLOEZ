import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import tkinter as tk
from tkinter import messagebox

class PreprocessGUI(QtWidgets.QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Image Preprocessing GUI")  # This is enough for full screen without warnings

        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.post_processed = self.gray.copy()

        self.blur_kernel_size = 0
        self.clip_limit = 2.0
        self.tile_grid_size = 8

        self.init_ui()
        self.update_image()
        
        self.showMaximized()
        
    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)
        
    def keyPressEvent(self, event: QtGui.QKeyEvent):
            if event.key() == QtCore.Qt.Key_Escape:
                sys.exit(0)  # quit python script entirely
            else:
                super().keyPressEvent(event)

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout()

        # Image layout using splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Original Image layout (label + image)
        original_container = QtWidgets.QWidget()
        original_layout = QtWidgets.QVBoxLayout(original_container)
        original_label_text = QtWidgets.QLabel("Original Image")
        original_label_text.setAlignment(QtCore.Qt.AlignCenter)
        original_label_text.setFont(QtGui.QFont("", 14, QtGui.QFont.Bold))
        self.original_label = QtWidgets.QLabel()
        self.original_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.original_label.setMinimumSize(600, 600)
        original_layout.addWidget(original_label_text)
        original_layout.addWidget(self.original_label)

        # Post-Processed Image layout (label + image)
        processed_container = QtWidgets.QWidget()
        processed_layout = QtWidgets.QVBoxLayout(processed_container)
        processed_label_text = QtWidgets.QLabel("Post-Processed Image")
        processed_label_text.setAlignment(QtCore.Qt.AlignCenter)
        processed_label_text.setFont(QtGui.QFont("", 14, QtGui.QFont.Bold))
        self.processed_label = QtWidgets.QLabel()
        self.processed_label.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.processed_label.setMinimumSize(600, 600)
        processed_layout.addWidget(processed_label_text)
        processed_layout.addWidget(self.processed_label)

        splitter.addWidget(original_container)
        splitter.addWidget(processed_container)
        splitter.setSizes([1, 1])

        main_layout.addWidget(splitter)

        # Controls
        controls_layout = QtWidgets.QVBoxLayout()

        # Create a big font
        big_font = QtGui.QFont()
        big_font.setPointSize(16)

        # Blur kernel
        blur_layout = QtWidgets.QHBoxLayout()
        blur_label = QtWidgets.QLabel("Blur Kernel Size")
        blur_label.setFont(big_font)
        blur_label_container = self.make_label_with_tooltip(
            "Blur Kernel Size",
            "Sets the size of the Gaussian blur kernel.\nLarger values increase smoothing to reduce noise,\nbut may blur important details.",
            big_font
        )
        
        self.blur_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blur_slider.setRange(0, 31)
        self.blur_slider.setSingleStep(2)
        self.blur_slider.setPageStep(2)
        self.blur_slider.setValue(self.blur_kernel_size)
        self.blur_slider.setFixedHeight(40)

        self.blur_spin = QtWidgets.QSpinBox()
        self.blur_spin.setRange(0, 31)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(self.blur_kernel_size)
        self.blur_spin.setFont(big_font)
        self.blur_spin.setFixedHeight(40)
        blur_layout.addWidget(blur_label_container)
        blur_layout.addWidget(self.blur_slider)
        #blur_layout.addStretch(1)
        blur_layout.addWidget(self.blur_spin)

        controls_layout.addLayout(blur_layout)

        # Clip limit
        clip_layout = QtWidgets.QHBoxLayout()
        clip_label = QtWidgets.QLabel("CLAHE Clip Limit")
        clip_label.setFont(big_font)
        clip_label_container = self.make_label_with_tooltip(
            "CLAHE Clip Limit",
            "Controls the contrast limit for CLAHE (Contrast Limited Adaptive Histogram Equalization).\nHigher values increase local contrast, enhancing details but potentially amplifying noise.",
            big_font
        )
        clip_layout.addWidget(clip_label_container)

        self.clip_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.clip_slider.setRange(1, 1000)
        self.clip_slider.setValue(int(self.clip_limit * 100))
        self.clip_slider.setFixedHeight(40)
        clip_layout.addWidget(self.clip_slider)

        self.clip_spin = QtWidgets.QDoubleSpinBox()
        self.clip_spin.setRange(0.01, 10.0)
        self.clip_spin.setSingleStep(0.01)
        self.clip_spin.setDecimals(2)
        self.clip_spin.setValue(self.clip_limit)
        self.clip_spin.setFont(big_font)
        self.clip_spin.setFixedHeight(40)
        clip_layout.addWidget(self.clip_spin)

        controls_layout.addLayout(clip_layout)

        # Tile grid size
        tile_layout = QtWidgets.QHBoxLayout()
        tile_label = QtWidgets.QLabel("CLAHE Tile Grid Size")
        tile_label.setFont(big_font)
        tile_label_container = self.make_label_with_tooltip(
            "CLAHE Tile Grid Size",
            "Sets the size of the grid for CLAHE.\nSmaller grid sizes enhance local contrast more locally,\nwhile larger grid sizes smooth contrast over bigger areas.",
            big_font
        )
        tile_layout.addWidget(tile_label_container)

        self.tile_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.tile_slider.setRange(1, 20)
        self.tile_slider.setValue(self.tile_grid_size)
        self.tile_slider.setFixedHeight(40)
        tile_layout.addWidget(self.tile_slider)

        self.tile_spin = QtWidgets.QSpinBox()
        self.tile_spin.setRange(1, 20)
        self.tile_spin.setValue(self.tile_grid_size)
        self.tile_spin.setFont(big_font)
        self.tile_spin.setFixedHeight(40)
        tile_layout.addWidget(self.tile_spin)

        controls_layout.addLayout(tile_layout)

        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)

        # Connect signals (your existing code)
        self.blur_slider.valueChanged.connect(self.update_from_sliders)
        self.clip_slider.valueChanged.connect(self.update_from_sliders)
        self.tile_slider.valueChanged.connect(self.update_from_sliders)

        self.blur_spin.valueChanged.connect(self.update_from_spins)
        self.clip_spin.valueChanged.connect(self.update_from_spins)
        self.tile_spin.valueChanged.connect(self.update_from_spins)

        # Update images on resize (your existing code)
        self.original_label.resizeEvent = self.resize_event_wrapper(self.original_label)
        self.processed_label.resizeEvent = self.resize_event_wrapper(self.processed_label)
        
    def make_label_with_tooltip(self, text, tooltip, font=None):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QtWidgets.QLabel(text)
        if font:
            label.setFont(font)
        label.setToolTip(tooltip)

        icon_label = QtWidgets.QLabel()
        icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
        icon_label.setPixmap(icon_pix.pixmap(14, 14))
        icon_label.setToolTip(tooltip)
        if font:
            icon_label.setFont(font)

        layout.addWidget(label)
        layout.addWidget(icon_label)
        # REMOVE layout.addStretch() here!

        return container

    def resize_event_wrapper(self, label):
        def resize_event(event):
            self.update_image()
            QtWidgets.QLabel.resizeEvent(label, event)
        return resize_event

    def update_from_sliders(self):
        blur_value = self.blur_slider.value()
        if blur_value % 2 == 0:
            blur_value += 1

        clip_value = self.clip_slider.value() / 100.0
        tile_value = self.tile_slider.value()

        self.blur_spin.blockSignals(True)
        self.clip_spin.blockSignals(True)
        self.tile_spin.blockSignals(True)

        self.blur_spin.setValue(blur_value)
        self.clip_spin.setValue(clip_value)
        self.tile_spin.setValue(tile_value)

        self.blur_spin.blockSignals(False)
        self.clip_spin.blockSignals(False)
        self.tile_spin.blockSignals(False)

        self.blur_kernel_size = blur_value
        self.clip_limit = clip_value
        self.tile_grid_size = tile_value

        self.update_image()

    def update_from_spins(self):
        blur_value = self.blur_spin.value()
        if blur_value % 2 == 0:
            blur_value += 1

        clip_value = self.clip_spin.value()
        tile_value = self.tile_spin.value()

        self.blur_slider.blockSignals(True)
        self.clip_slider.blockSignals(True)
        self.tile_slider.blockSignals(True)

        self.blur_slider.setValue(blur_value)
        self.clip_slider.setValue(int(clip_value * 100))
        self.tile_slider.setValue(tile_value)

        self.blur_slider.blockSignals(False)
        self.clip_slider.blockSignals(False)
        self.tile_slider.blockSignals(False)

        self.blur_kernel_size = blur_value
        self.clip_limit = clip_value
        self.tile_grid_size = tile_value

        self.update_image()

    def update_image(self):
        post_processed = self.post_processed

        if self.clip_limit and self.tile_grid_size:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_grid_size, self.tile_grid_size))
            post_processed = clahe.apply(self.gray)

        if self.blur_kernel_size > 0:
            post_processed = cv2.GaussianBlur(post_processed, (self.blur_kernel_size, self.blur_kernel_size), 0)

        original_bgr = self.image.copy()
        post_bgr = cv2.cvtColor(post_processed, cv2.COLOR_GRAY2BGR)

        self.display_image(self.original_label, original_bgr)
        self.display_image(self.processed_label, post_bgr)
        
    def create_label_with_tooltip(self, text, tooltip_text, font):
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(text)
        label.setFont(font)

        # Question mark icon as a QLabel with blue text and underline for clarity
        question_mark = QtWidgets.QLabel("?")
        question_mark.setFont(font)
        question_mark.setStyleSheet("color: blue; text-decoration: underline;")
        question_mark.setToolTip(tooltip_text)
        question_mark.setFixedWidth(15)
        question_mark.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(label)
        layout.addWidget(question_mark)
        layout.addStretch(1)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        return container

    def display_image(self, label, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)

        # Scale pixmap to label's current size
        scaled_pixmap = pixmap.scaled(
            label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

def run_preprocess_gui(image_path, suppress_instructions=False):
    # Show instructions popup if not suppressed
    if not suppress_instructions:
        root = tk.Tk()
        root.title("Instructions")

        # Your instructions text
        instructions = (
            "Instructions for using the Image Preprocessing GUI:\n\n"
            "- Use the sliders and spin boxes to adjust blur kernel size, CLAHE clip limit, and tile grid size.\n"
            "- The original and post-processed images are shown side by side.\n"
            "- Press ESC to quit the application.\n"
            "- Adjust parameters to achieve the desired image enhancement.\n"
            "- When finished, close the window.\n"
        )

        label = tk.Label(root, text=instructions, justify="left", padx=20, pady=20, font=("Helvetica", 12))
        label.pack()

        # Center the window
        root.update_idletasks()  # Make sure geometry is calculated
        window_width = root.winfo_width()
        window_height = root.winfo_height()

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        root.geometry(f"+{x}+{y}")

        # Wait for user to press OK to continue
        ok_button = tk.Button(root, text="OK", command=root.destroy, padx=10, pady=5)
        ok_button.pack(pady=(0, 20))

        root.mainloop() 

    app = QtWidgets.QApplication.instance()
    owns_app = False
    if not app:
        app = QtWidgets.QApplication(sys.argv)
        owns_app = True

    gui = PreprocessGUI(image_path)

    result = {}

    def capture_results():
        result['original'] = gui.image
        result['gray'] = gui.gray
        result['post_processed'] = gui.post_processed

    app.aboutToQuit.connect(capture_results)

    app.exec_()

    if owns_app:
        app.quit()

    return result.get('original'), result.get('gray'), result.get('post_processed')