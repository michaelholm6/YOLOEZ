import sys
import os
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import tkinter as tk

os.environ["QT_LOGGING_RULES"] = "*.warning=false"

def detect_edges(blurred, crack_expansion, model_path='model.yml.gz',
                 confidence_threshold=0.15, aoi_pts=None,
                 max_dim=3000):
    h, w = blurred.shape[:2]
    scale_factor = min(1.0, max_dim / max(h, w))

    if scale_factor < 1.0:
        blurred = cv2.resize(blurred, None, fx=scale_factor, fy=scale_factor,
                             interpolation=cv2.INTER_AREA)
        if aoi_pts is not None:
            aoi_pts = [(int(x * scale_factor), int(y * scale_factor)) for (x, y) in aoi_pts]

    if getattr(sys, 'frozen', False):
        model_path = os.path.join(sys._MEIPASS, model_path)

    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)
    blurred_float = blurred.astype(np.float32) / 255.0
    if len(blurred_float.shape) == 2:
        blurred_float = cv2.cvtColor(blurred_float, cv2.COLOR_GRAY2BGR)

    edges = edge_detector.detectEdges(blurred_float)
    filtered_edges = (edges > confidence_threshold).astype(np.uint8) * 255

    kernel = np.ones((crack_expansion, crack_expansion), np.uint8)
    dilated = cv2.dilate(filtered_edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    if aoi_pts is not None:
        aoimask = np.zeros(eroded.shape[:2], dtype=np.uint8)
        polygon_np = np.array(aoi_pts, np.int32)
        cv2.fillPoly(aoimask, [polygon_np], 255)
        eroded = cv2.bitwise_and(eroded, aoimask)

    return eroded, scale_factor


class ClickableSlider(QtWidgets.QSlider):
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            pos_ratio = event.x() / self.width()
            pos_ratio = max(0.0, min(1.0, pos_ratio))
            value_range = self.maximum() - self.minimum()
            new_val = round(self.minimum() + pos_ratio * value_range)
            self.setValue(new_val)
            event.accept()
        super().mousePressEvent(event)


class CrackDetectionGUI(QtWidgets.QWidget):
    def __init__(self, original_image, blurred_image, area_of_interest_pts, model_path='model.yml.gz'):
        super().__init__()

        self.original_image = original_image.copy()
        self.blurred_image = blurred_image
        self.model_path = model_path
        self.area_of_interest_pts = area_of_interest_pts

        self.confidence_threshold = 0.15
        self.crack_expansion = 3
        self.line_thickness = 2

        self.last_applied_confidence = self.confidence_threshold
        self.last_applied_expansion = self.crack_expansion
        self.last_applied_thickness = self.line_thickness

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Sliders
        self.conf_slider = ClickableSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(self.confidence_threshold * 100))
        self.conf_slider.setFixedHeight(40)
        self.conf_slider.valueChanged.connect(self.update_from_sliders)

        self.crack_slider = ClickableSlider(QtCore.Qt.Horizontal)
        self.crack_slider.setMinimum(1)
        self.crack_slider.setMaximum(20)
        self.crack_slider.setValue(self.crack_expansion)
        self.crack_slider.setFixedHeight(40)
        self.crack_slider.valueChanged.connect(self.update_from_sliders)

        self.thickness_slider = ClickableSlider(QtCore.Qt.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(50)
        self.thickness_slider.setValue(self.line_thickness)
        self.thickness_slider.setFixedHeight(40)
        self.thickness_slider.valueChanged.connect(self.update_from_sliders)

        # Spin boxes
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setDecimals(2)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(self.confidence_threshold)
        self.conf_spin.editingFinished.connect(self.update_from_spin_boxes)

        self.crack_spin = QtWidgets.QSpinBox()
        self.crack_spin.setRange(0, 50)
        self.crack_spin.setValue(self.crack_expansion)
        self.crack_spin.editingFinished.connect(self.update_from_spin_boxes)

        self.thickness_spin = QtWidgets.QSpinBox()
        self.thickness_spin.setRange(1, 50)
        self.thickness_spin.setValue(self.line_thickness)
        self.thickness_spin.editingFinished.connect(self.update_from_spin_boxes)

        # Labels with tooltips
        self.conf_label_container = self.make_label_with_tooltip(
            "Confidence:",
            "Sets the confidence threshold for edge detection.\n"
            "Higher values include only stronger edges,\n"
            "lower values include more edges and noise."
        )
        self.crack_label_container = self.make_label_with_tooltip(
            "Crack expansion:",
            "Controls morphological expansion of cracks.\n"
            "Higher values connect nearby edges.\n"
        )
        self.thickness_label_container = self.make_label_with_tooltip(
            "Line thickness:",
            "Controls the thickness of the contour lines drawn on the image."
        )

        # Layouts
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)

        conf_layout = QtWidgets.QHBoxLayout()
        conf_layout.addWidget(self.conf_label_container)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_spin)

        crack_layout = QtWidgets.QHBoxLayout()
        crack_layout.addWidget(self.crack_label_container)
        crack_layout.addWidget(self.crack_slider)
        crack_layout.addWidget(self.crack_spin)

        thickness_layout = QtWidgets.QHBoxLayout()
        thickness_layout.addWidget(self.thickness_label_container)
        thickness_layout.addWidget(self.thickness_slider)
        thickness_layout.addWidget(self.thickness_spin)

        layout.addLayout(conf_layout)
        layout.addLayout(crack_layout)
        layout.addLayout(thickness_layout)

        self.setLayout(layout)
        self.setWindowTitle("Crack Detection GUI")

        self.final_contours = []

        self.showMaximized()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Escape:
            sys.exit(0)
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.update_image()

    def make_label_with_tooltip(self, text, tooltip_text):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        font = QtGui.QFont()
        font.setPointSize(16)

        label = QtWidgets.QLabel(text)
        label.setFont(font)
        label.setToolTip(tooltip_text)

        icon_label = QtWidgets.QLabel()
        icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
        icon_label.setPixmap(icon_pix.pixmap(16, 16))
        icon_label.setToolTip(tooltip_text)

        layout.addWidget(label)
        layout.addWidget(icon_label)
        return container

    def update_from_spin_boxes(self):
        new_conf = self.conf_spin.value()
        new_exp = self.crack_spin.value()
        new_thick = self.thickness_spin.value()

        if (
            np.isclose(new_conf, self.last_applied_confidence, atol=1e-6)
            and new_exp == self.last_applied_expansion
            and new_thick == self.last_applied_thickness
        ):
            return

        self.confidence_threshold = new_conf
        self.crack_expansion = new_exp
        self.line_thickness = new_thick

        self.conf_slider.blockSignals(True)
        self.crack_slider.blockSignals(True)
        self.thickness_slider.blockSignals(True)
        self.conf_slider.setValue(int(self.confidence_threshold * 100))
        self.crack_slider.setValue(self.crack_expansion)
        self.thickness_slider.setValue(self.line_thickness)
        self.conf_slider.blockSignals(False)
        self.crack_slider.blockSignals(False)
        self.thickness_slider.blockSignals(False)

        self.update_image()

        self.last_applied_confidence = self.confidence_threshold
        self.last_applied_expansion = self.crack_expansion
        self.last_applied_thickness = self.line_thickness

    def update_from_sliders(self):
        raw_value = self.conf_slider.value() / 100.0
        self.confidence_threshold = max(0.01, raw_value)
        self.crack_expansion = self.crack_slider.value()
        self.line_thickness = self.thickness_slider.value()

        self.conf_spin.blockSignals(True)
        self.crack_spin.blockSignals(True)
        self.thickness_spin.blockSignals(True)
        self.conf_spin.setValue(self.confidence_threshold)
        self.crack_spin.setValue(self.crack_expansion)
        self.thickness_spin.setValue(self.line_thickness)
        self.conf_spin.blockSignals(False)
        self.crack_spin.blockSignals(False)
        self.thickness_spin.blockSignals(False)

        self.update_image()

        self.last_applied_confidence = self.confidence_threshold
        self.last_applied_expansion = self.crack_expansion
        self.last_applied_thickness = self.line_thickness

    def update_image(self):
        edges, scale_factor = detect_edges(
            self.blurred_image,
            self.crack_expansion,
            self.model_path,
            self.confidence_threshold,
            aoi_pts=self.area_of_interest_pts,
            max_dim=1200
        )

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if scale_factor < 1.0:
            scaled_contours = []
            for cnt in contours:
                cnt = (cnt.astype(np.float32) / scale_factor).astype(np.int32)
                scaled_contours.append(cnt)
            contours = scaled_contours

        self.final_contours = contours

        display_img = self.original_image.copy()
        if self.area_of_interest_pts:
            cv2.polylines(display_img, [np.array(self.area_of_interest_pts, np.int32)],
                          True, (0, 255, 255), self.line_thickness)
        cv2.drawContours(display_img, contours, -1, (0, 0, 255), self.line_thickness)

        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = display_img_rgb.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(display_img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)

        label_width = self.image_label.width()
        label_height = self.image_label.height()
        scaled_pixmap = pixmap.scaled(label_width, label_height,
                                      QtCore.Qt.KeepAspectRatio,
                                      QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)

    def get_final_results(self):
        return self.final_contours, self.line_thickness


def detect_cracks(original_image, blurred_image, area_of_interest_pts, suppress_instructions=False):
    if not suppress_instructions:
        root = tk.Tk()
        root.title("Instructions")
        instructions = (
            "Use the sliders or spin boxes to adjust:\n"
            "- Confidence threshold: higher filters out weak edges.\n"
            "- Crack expansion: controls crack thickness and connectivity.\n"
            "- Line thickness: changes contour line width.\n"
            "The image updates in real-time.\n"
            "Press ESC to exit and close the application.\n"
            "Close the window when done.\n"
        )
        label = tk.Label(root, text=instructions, justify="left", padx=20, pady=20, font=("Helvetica", 12))
        label.pack()
        root.update_idletasks()
        window_width = root.winfo_width()
        window_height = root.winfo_height()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        root.geometry(f"+{x}+{y}")
        ok_button = tk.Button(root, text="OK", command=root.destroy, padx=10, pady=5)
        ok_button.pack(pady=(0, 20))
        root.mainloop()

    app = QtWidgets.QApplication.instance()
    close_app_after = False

    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        close_app_after = True

    gui = CrackDetectionGUI(original_image, blurred_image, area_of_interest_pts)
    gui.show()
    app.exec_()
    contours, line_thickness = gui.get_final_results()

    if close_app_after:
        app.quit()

    return contours, line_thickness