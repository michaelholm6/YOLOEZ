import sys
import os
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import tkinter as tk
from DefectDetect.labelling_workflow.clip_cracks_to_area_of_interest import clip_edges_to_polygon
from utils import make_label_with_tooltip

os.environ["QT_LOGGING_RULES"] = "*.warning=false"

def detect_edges(blurred, model_path='model.yml.gz',
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

    if aoi_pts is not None:
        filtered_edges = clip_edges_to_polygon(filtered_edges, aoi_pts)

    return filtered_edges, scale_factor


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
        self.line_thickness = 2

        self.last_applied_confidence = self.confidence_threshold
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

        self.thickness_spin = QtWidgets.QSpinBox()
        self.thickness_spin.setRange(1, 50)
        self.thickness_spin.setValue(self.line_thickness)
        self.thickness_spin.editingFinished.connect(self.update_from_spin_boxes)

        # Labels with tooltips
        self.conf_label_container = make_label_with_tooltip(self, 
            "Confidence:",
            "Sets the confidence threshold for edge detection.\n"
            "Higher values include only stronger edges,\n"
            "lower values include more edges and noise."
        )
        self.thickness_label_container = make_label_with_tooltip(self,
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

        thickness_layout = QtWidgets.QHBoxLayout()
        thickness_layout.addWidget(self.thickness_label_container)
        thickness_layout.addWidget(self.thickness_slider)
        thickness_layout.addWidget(self.thickness_spin)

        layout.addLayout(conf_layout)
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

    def update_from_spin_boxes(self):
        new_conf = self.conf_spin.value()
        new_thick = self.thickness_spin.value()

        if (
            np.isclose(new_conf, self.last_applied_confidence, atol=1e-6)
            and new_thick == self.last_applied_thickness
        ):
            return

        self.confidence_threshold = new_conf
        self.line_thickness = new_thick

        self.conf_slider.blockSignals(True)
        self.thickness_slider.blockSignals(True)
        self.conf_slider.setValue(int(self.confidence_threshold * 100))
        self.thickness_slider.setValue(self.line_thickness)
        self.conf_slider.blockSignals(False)
        self.thickness_slider.blockSignals(False)

        self.update_image()

        self.last_applied_confidence = self.confidence_threshold
        self.last_applied_thickness = self.line_thickness

    def update_from_sliders(self):
        raw_value = self.conf_slider.value() / 100.0
        self.confidence_threshold = max(0.01, raw_value)
        self.line_thickness = self.thickness_slider.value()

        self.conf_spin.blockSignals(True)
        self.thickness_spin.blockSignals(True)
        self.conf_spin.setValue(self.confidence_threshold)
        self.thickness_spin.setValue(self.line_thickness)
        self.conf_spin.blockSignals(False)
        self.thickness_spin.blockSignals(False)

        self.update_image()

        self.last_applied_confidence = self.confidence_threshold
        self.last_applied_thickness = self.line_thickness

    def update_image(self):
        contours, scale_factor = detect_edges(
            self.blurred_image,
            self.model_path,
            self.confidence_threshold,
            aoi_pts=self.area_of_interest_pts,
            max_dim=1200
        )

        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if scale_factor < 1.0:
            scaled_contours = []
            for cnt in contours:
                cnt = (cnt.astype(np.float32) / scale_factor).astype(np.int32)
                scaled_contours.append(cnt)
            contours = scaled_contours

        self.final_contours = contours

        display_img = self.original_image.copy()
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


def detect_cracks(original_image, blurred_image, area_of_interest_pts):

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