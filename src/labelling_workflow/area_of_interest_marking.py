# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QSlider
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QKeyEvent, QImage
from PyQt5.QtCore import Qt, QPoint, QTimer
import cv2
from utils import show_error_window


def get_polygon_from_user(image, app):
    class PolygonDrawer(QWidget):
        def __init__(self, image):
            super().__init__()
            self.orig_image = image
            self.orig_height, self.orig_width = image.shape[:2]
            self.polygon_points = []
            self.polygon_closed = False
            self.setWindowTitle("Draw Polygon - C: close, R: reset, S: skip")
            self.showMaximized()
            self.setMouseTracking(True)

        def resizeEvent(self, event):
            self.update_scaled_pixmap()

        def update_scaled_pixmap(self):
            window_size = self.size()
            qimg = self.numpy_to_qpixmap(self.orig_image)
            self.scaled_pixmap = qimg.scaled(window_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.offset_x = (self.width() - self.scaled_pixmap.width()) // 2
            self.offset_y = (self.height() - self.scaled_pixmap.height()) // 2
            self.scale_x = self.scaled_pixmap.width() / self.orig_width
            self.scale_y = self.scaled_pixmap.height() / self.orig_height
            self.update()

        def numpy_to_qpixmap(self, img):
            if len(img.shape) == 2:
                qimg = QPixmap(QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8))
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QPixmap(QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format_RGB888))
            return qimg

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(255, 255, 255))
            painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)

            if len(self.polygon_points) > 0:
                pen = QPen(QColor(0, 255, 0), 5)
                painter.setPen(pen)
                for i in range(len(self.polygon_points) - 1):
                    painter.drawLine(self.polygon_points[i], self.polygon_points[i + 1])
                if self.polygon_closed:
                    painter.drawLine(self.polygon_points[-1], self.polygon_points[0])

                pen = QPen(QColor(255, 0, 0))
                painter.setPen(pen)
                painter.setBrush(QColor(255, 0, 0))
                for pt in self.polygon_points:
                    painter.drawEllipse(pt, 6, 6)

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                pos = event.pos()
                if (
                    self.offset_x <= pos.x() <= self.offset_x + self.scaled_pixmap.width()
                    and self.offset_y <= pos.y() <= self.offset_y + self.scaled_pixmap.height()
                ):
                    self.polygon_points.append(QPoint(pos.x(), pos.y()))
                    self.polygon_closed = False
                    self.update()

        def keyPressEvent(self, event: QKeyEvent):
            key = event.key()
            if key == Qt.Key_C:
                if len(self.polygon_points) > 2:
                    self.polygon_closed = True
                    self.update()
                    QTimer.singleShot(300, self.close)
                else:
                    print("Need at least 3 points to form a polygon.")
            elif key == Qt.Key_R:
                self.polygon_points = []
                self.polygon_closed = False
                self.update()
            elif key == Qt.Key_S:
                self.polygon_points = []
                self.polygon_closed = False
                self.close()

    drawer = PolygonDrawer(image)
    drawer.update_scaled_pixmap()
    drawer.show()
    app.exec_()

    if drawer.polygon_closed and len(drawer.polygon_points) > 2:
        scaled_points = []
        for pt in drawer.polygon_points:
            x = int((pt.x() - drawer.offset_x) / drawer.scale_x)
            y = int((pt.y() - drawer.offset_y) / drawer.scale_y)
            scaled_points.append([x, y])
        return scaled_points
    else:
        # fallback: full image rectangle
        h, w = image.shape[:2]
        return [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]

import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QImage, QKeyEvent
)
from PyQt5.QtCore import Qt, QPoint


class PolygonCanvas(QWidget):
    """
    Widget that draws an image scaled to the widget (keeping aspect ratio)
    and lets the user click to add polygon vertices. Internally stores/display
    points in widget coordinates. Provides conversions to/from image coords.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.orig_image = None  # BGR numpy array
        self.scaled_pixmap = None  # QPixmap
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0  # uniform scale (scaled_width / orig_width)
        self.orig_w = 1
        self.orig_h = 1

        self.polygon_pts_widget = []  # list of QPoint in widget coordinates
        self.polygon_closed = False
        self.line_thickness = 2

        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)

    # --- image loading / scaling helpers ---
    def load_image(self, image_bgr, saved_polygon_image_coords=None):
        """image_bgr: numpy BGR image. saved_polygon_image_coords: list of [x,y] in image coords"""
        self.orig_image = image_bgr
        self.orig_h, self.orig_w = image_bgr.shape[:2]
        self.update_scaled_image()

        # If there's a saved polygon in image coords, convert to widget coords now
        self.polygon_pts_widget = []
        self.polygon_closed = False
        if saved_polygon_image_coords:
            for (ix, iy) in saved_polygon_image_coords:
                wx = int(ix * self.scale + self.offset_x)
                wy = int(iy * self.scale + self.offset_y)
                self.polygon_pts_widget.append(QPoint(wx, wy))
            if len(self.polygon_pts_widget) > 2:
                self.polygon_closed = True
        self.update()

    def update_scaled_image(self):
        """Compute scaled_pixmap, offsets and scale factor based on current widget size."""
        if self.orig_image is None:
            return
        h_win = max(1, self.height())
        w_win = max(1, self.width())

        # convert to QImage/QPixmap
        if len(self.orig_image.shape) == 2:
            qimg = QImage(self.orig_image.data, self.orig_image.shape[1], self.orig_image.shape[0],
                          self.orig_image.strides[0], QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        # scale with keep aspect ratio
        scaled = pix.scaled(w_win, h_win, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scaled_pixmap = scaled

        # offsets to center
        self.offset_x = (w_win - scaled.width()) // 2
        self.offset_y = (h_win - scaled.height()) // 2

        # uniform scale from image -> widget
        self.scale = scaled.width() / max(1, self.orig_w)

    # --- coordinate conversions ---
    def widget_to_image_coords(self, qpoint):
        """Convert a QPoint (widget coords) to image (x, y) ints."""
        xw, yw = qpoint.x(), qpoint.y()
        ix = int(round((xw - self.offset_x) / max(1e-6, self.scale)))
        iy = int(round((yw - self.offset_y) / max(1e-6, self.scale)))
        # clamp
        ix = max(0, min(self.orig_w - 1, ix))
        iy = max(0, min(self.orig_h - 1, iy))
        return ix, iy

    def image_to_widget_coords(self, ix, iy):
        """Convert image coords to widget QPoint."""
        wx = int(round(ix * self.scale + self.offset_x))
        wy = int(round(iy * self.scale + self.offset_y))
        return QPoint(wx, wy)

    # --- painting / interaction ---
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))  # white background

        if self.scaled_pixmap:
            painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)

        if len(self.polygon_pts_widget) > 0:
            pen = QPen(QColor(0, 200, 0), self.line_thickness)
            painter.setPen(pen)
            for i in range(len(self.polygon_pts_widget) - 1):
                painter.drawLine(self.polygon_pts_widget[i], self.polygon_pts_widget[i + 1])
            if self.polygon_closed:
                painter.drawLine(self.polygon_pts_widget[-1], self.polygon_pts_widget[0])

            # draw nodes
            pen = QPen(QColor(200, 0, 0))
            painter.setPen(pen)
            painter.setBrush(QColor(200, 0, 0))
            for pt in self.polygon_pts_widget:
                painter.drawEllipse(pt, 5, 5)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.scaled_pixmap:
            pos = event.pos()
            # ensure click is on the displayed image area
            if (self.offset_x <= pos.x() <= self.offset_x + self.scaled_pixmap.width() and
                    self.offset_y <= pos.y() <= self.offset_y + self.scaled_pixmap.height()):
                self.polygon_pts_widget.append(pos)
                self.polygon_closed = False
                self.update()

    def resizeEvent(self, event):
        # Recompute scaled image and convert any existing polygon in image coords back
        prev_image = self.orig_image
        if prev_image is not None:
            # Convert current stored polygon (widget coords) into image coords,
            # then reload and reconvert to widget coords under new scale.
            img_coords = [self.widget_to_image_coords(pt) for pt in self.polygon_pts_widget]
            self.update_scaled_image()
            self.polygon_pts_widget = [self.image_to_widget_coords(ix, iy) for (ix, iy) in img_coords]
        else:
            self.update_scaled_image()
        super().resizeEvent(event)

    # --- utility to get polygon in image coords ---
    def get_polygon_image_coords(self):
        if len(self.polygon_pts_widget) == 0:
            return []
        pts_img = [self.widget_to_image_coords(pt) for pt in self.polygon_pts_widget]
        return pts_img

    def reset_polygon(self):
        self.polygon_pts_widget = []
        self.polygon_closed = False
        self.update()

    def close_polygon(self):
        if len(self.polygon_pts_widget) > 2:
            self.polygon_closed = True
            self.update()
            return True
        return False


class PolygonAnnotatorWindow(QWidget):
    """
    Main window: top area is the PolygonCanvas, bottombar contains Previous/Next buttons.
    """

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.index = 0
        self.results = {p: [] for p in image_paths}  # polygons in image coords

        # Canvas
        self.canvas = PolygonCanvas(self)

        # Buttons
        self.prev_btn = QPushButton("◀ Previous")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(lambda: self.change_image(-1))
        self.next_btn.clicked.connect(lambda: self.change_image(1))

        # Layout buttons full width
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)

        # Status label
        self.status_label = QLabel("")
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(50)
        self.thickness_slider.setValue(self.canvas.line_thickness)
        self.thickness_slider.setTickPosition(QSlider.TicksBelow)
        self.thickness_slider.setTickInterval(1)
        self.thickness_slider.valueChanged.connect(self.update_line_thickness)

        self.thickness_value_label = QLabel(str(self.canvas.line_thickness))
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Line Thickness:"))
        slider_layout.addWidget(self.thickness_slider)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addLayout(slider_layout)
        main_layout.addLayout(status_layout)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("AOI Annotator (C=close polygon, R=reset)")
        self.showMaximized()
        self.showMaximized()  # show first
        self.raise_()
        self.activateWindow()

        # Load first image
        self.load_current_image()

    def update_line_thickness(self, value):
        self.canvas.line_thickness = value
        self.canvas.update()

    def load_current_image(self):
        path = self.image_paths[self.index]
        img = cv2.imread(path)
        if img is None:
            show_error_window(f"Failed to load image: {path}")
            return
        saved = self.results.get(path, [])
        self.canvas.load_image(img, saved_polygon_image_coords=saved)
        self.update_status_label()
        self.update_button_states()

    def update_status_label(self):
        self.status_label.setText(f"Image {self.index + 1} / {len(self.image_paths)}")

    def update_button_states(self):
        self.prev_btn.setEnabled(self.index > 0)
        self.next_btn.setEnabled(self.index < len(self.image_paths) - 1)

    def save_current_polygon(self):
        path = self.image_paths[self.index]
        self.results[path] = self.canvas.get_polygon_image_coords()

    def change_image(self, delta):
        self.save_current_polygon()
        self.index = max(0, min(len(self.image_paths) - 1, self.index + delta))
        self.load_current_image()

    def keyPressEvent(self, event: QKeyEvent):
        k = event.key()
        if k == Qt.Key_C:
            closed = self.canvas.close_polygon()
            if closed:
                self.save_current_polygon()
        elif k == Qt.Key_R:
            self.canvas.reset_polygon()
        elif k == Qt.Key_Right:
            self.change_image(1)
        elif k == Qt.Key_Left:
            self.change_image(-1)
        elif k == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.save_current_polygon()
        event.accept()
        
        

def annotate_images(image_paths):
    """Top-level helper: launches the GUI and returns {path: [[x,y], ...]}, line thickness"""
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    win = PolygonAnnotatorWindow(image_paths)
    win.show()
    if created_app:
        app.exec_()
    else:
        app.exec_()

    # return polygons and final line thickness
    return win.results, win.canvas.line_thickness


# Example usage
if __name__ == "__main__":
    folder = "path/to/images"
    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    image_files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(valid_exts)]

    polygons = annotate_images(image_files)
    for path, poly in polygons.items():
        print(path, poly)
