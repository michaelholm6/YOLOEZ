# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

from utils import show_error_window

import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider,
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QKeyEvent
from PyQt5.QtCore import Qt, QPoint, QEventLoop


class PolygonCanvas(QWidget):
    """
    Widget that draws an image scaled to the widget (keeping aspect ratio)
    and lets the user click to add polygon vertices. Internally stores/display
    points in widget coordinates. Provides conversions to/from image coords.
    """

    def __init__(self, parent=None):
        """Initialize an empty canvas; call load_image() to display an image."""
        super().__init__(parent)
        self.orig_image = None  # BGR numpy array
        self.scaled_pixmap = None  # QPixmap
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0  # uniform scale (scaled_width / orig_width)
        self.orig_w = 1
        self.orig_h = 1
        self.polygons = []  # list of {"points": [QPoint,...], "closed": bool}
        self.current_polygon = {"points": [], "closed": False}
        self.redo_stack = []
        self.dragging = False
        self.last_drag_point = None

        self.polygon_pts_widget = []  # list of QPoint in widget coordinates
        self.polygon_closed = False
        self.line_thickness = 2

        self.hover_first = False

        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)

    # --- image loading / scaling helpers ---
    def load_image(self, image_bgr, saved_polygon_image_coords=None):
        """Load a BGR numpy image and optionally restore previously saved polygons.

        Args:
            image_bgr: The image to display.
            saved_polygon_image_coords: List of polygons in image coordinates to pre-populate.
        """
        self.orig_image = image_bgr
        self.orig_h, self.orig_w = image_bgr.shape[:2]
        self.update_scaled_image()

        self.polygons = []
        self.current_polygon = {"points": [], "closed": False}
        self.hover_first = False

        if saved_polygon_image_coords:
            for poly_img in saved_polygon_image_coords:
                poly_widget = {
                    "points": [
                        self.image_to_widget_coords(ix, iy) for (ix, iy) in poly_img
                    ],
                    "closed": True,
                }
                self.polygons.append(poly_widget)

        self.update()

    def update_scaled_image(self):
        """Compute scaled_pixmap, offsets and scale factor based on current widget size."""
        if self.orig_image is None:
            return
        h_win = max(1, self.height())
        w_win = max(1, self.width())

        if len(self.orig_image.shape) == 2:
            qimg = QImage(
                self.orig_image.data,
                self.orig_image.shape[1],
                self.orig_image.shape[0],
                self.orig_image.strides[0],
                QImage.Format_Grayscale8,
            )
        else:
            rgb = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format_RGB888,
            )

        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(w_win, h_win, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scaled_pixmap = scaled

        self.offset_x = (w_win - scaled.width()) // 2
        self.offset_y = (h_win - scaled.height()) // 2
        self.scale = scaled.width() / max(1, self.orig_w)

    def mouseMoveEvent(self, event):
        """Track hover proximity to the first point and accumulate drag-drawn points."""
        pos = event.pos()
        pts = self.current_polygon["points"]

        if len(pts) > 0 and not self.current_polygon["closed"]:
            first_pt = pts[0]
            dist_sq = (first_pt.x() - pos.x()) ** 2 + (first_pt.y() - pos.y()) ** 2
            self.hover_first = dist_sq <= 100
        else:
            if self.hover_first:
                self.hover_first = False

        if self.dragging:
            if (
                self.last_drag_point is None
                or (pos - self.last_drag_point).manhattanLength() > 15
            ):
                pts.append(pos)
                self.last_drag_point = pos

        self.update()

    def widget_to_image_coords(self, qpoint):
        """Convert a QPoint (widget coords) to image (x, y) ints."""
        xw, yw = qpoint.x(), qpoint.y()
        ix = int(round((xw - self.offset_x) / max(1e-6, self.scale)))
        iy = int(round((yw - self.offset_y) / max(1e-6, self.scale)))
        ix = max(0, min(self.orig_w - 1, ix))
        iy = max(0, min(self.orig_h - 1, iy))
        return ix, iy

    def image_to_widget_coords(self, ix, iy):
        """Convert image coords to widget QPoint."""
        wx = int(round(ix * self.scale + self.offset_x))
        wy = int(round(iy * self.scale + self.offset_y))
        return QPoint(wx, wy)

    def paintEvent(self, event):
        """Draw the scaled image and all polygon overlays."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        if self.scaled_pixmap:
            painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)

        for poly in self.polygons:
            pts = poly["points"]

            pen_lines = QPen(QColor(0, 200, 0), self.line_thickness)
            painter.setPen(pen_lines)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            if poly["closed"] and len(pts) > 2:
                painter.drawLine(pts[-1], pts[0])

            pen_points = QPen(QColor(200, 0, 0))
            painter.setPen(pen_points)
            painter.setBrush(QColor(200, 0, 0))
            for pt in pts:
                painter.drawEllipse(
                    pt, 2 * self.line_thickness, 2 * self.line_thickness
                )

        pts = self.current_polygon["points"]

        pen_lines = QPen(QColor(0, 200, 0), self.line_thickness)
        painter.setPen(pen_lines)
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])

        for i, pt in enumerate(pts):
            if i == 0 and self.hover_first and len(pts) >= 3:
                color = QColor(0, 0, 255)
            else:
                color = QColor(200, 0, 0)
            painter.setPen(QPen(color))
            painter.setBrush(color)
            painter.drawEllipse(pt, 2 * self.line_thickness, 2 * self.line_thickness)

    def mouseReleaseEvent(self, event):
        """On left release, append the drag endpoint or close the polygon if near the first point."""
        if event.button() != Qt.LeftButton or not self.scaled_pixmap:
            return

        pos = event.pos()
        self.dragging = False
        self.last_drag_point = None

        pts = self.current_polygon["points"]
        if not pts:
            return

        if len(pts) >= 3:
            first_pt = pts[0]
            dist_sq = (first_pt.x() - pos.x()) ** 2 + (first_pt.y() - pos.y()) ** 2
            if dist_sq <= 100:
                self.close_polygon()
                return
            else:
                pts.append(pos)

        self.update()

    def mousePressEvent(self, event):
        """Add a polygon vertex on left click, or close the polygon if near the first point."""
        if event.button() != Qt.LeftButton or not self.scaled_pixmap:
            return

        pos = event.pos()

        if not (
            self.offset_x <= pos.x() <= self.offset_x + self.scaled_pixmap.width()
            and self.offset_y <= pos.y() <= self.offset_y + self.scaled_pixmap.height()
        ):
            return

        pts = self.current_polygon["points"]
        self.dragging = True
        self.last_drag_point = pos

        if len(pts) >= 3:
            first_pt = pts[0]
            dist_sq = (first_pt.x() - pos.x()) ** 2 + (first_pt.y() - pos.y()) ** 2
            if dist_sq <= 100:  # within 10px
                self.close_polygon()
                return

        self.redo_stack.clear()
        pts.append(pos)
        self.update()

    def resizeEvent(self, event):
        """Re-scale the image and recompute widget coordinates for existing polygon points."""
        # Recompute scaled image and convert any existing polygon in image coords back
        prev_image = self.orig_image
        if prev_image is not None:
            # Convert current stored polygon (widget coords) into image coords,
            # then reload and reconvert to widget coords under new scale.
            img_coords = [
                self.widget_to_image_coords(pt) for pt in self.polygon_pts_widget
            ]
            self.update_scaled_image()
            self.polygon_pts_widget = [
                self.image_to_widget_coords(ix, iy) for (ix, iy) in img_coords
            ]
        else:
            self.update_scaled_image()
        super().resizeEvent(event)

    def get_polygon_image_coords(self):
        """Return all completed polygons as lists of (x, y) image-coordinate tuples."""
        all_polys = []
        for poly in self.polygons:
            pts_img = [self.widget_to_image_coords(pt) for pt in poly["points"]]
            all_polys.append(pts_img)
        return all_polys

    def close_polygon(self):
        """Finalize the in-progress polygon and add it to the completed list.  Returns True on success."""
        if len(self.current_polygon["points"]) > 2:
            self.current_polygon["closed"] = True
            self.polygons.append(self.current_polygon)

            self.redo_stack.clear()

            self.current_polygon = {"points": [], "closed": False}
            self.update()
            return True
        return False

    def undo(self):
        """Remove the last added point or cancel the last completed polygon."""
        if self.current_polygon["points"]:
            action = {"action": "cancel_current", "polygon": self.current_polygon}
            self.redo_stack.append(action)
            self.current_polygon = {"points": [], "closed": False}
            self.hover_first = False

        elif self.polygons:
            last = self.polygons.pop()
            self.redo_stack.append({"action": "remove", "polygon": last})

        self.update()

    def redo(self):
        """Re-apply the last undone polygon action."""
        if not self.redo_stack:
            return

        action = self.redo_stack.pop()

        if action["action"] == "remove":
            self.polygons.append(action["polygon"])

        elif action["action"] == "add":
            if self.polygons and self.polygons[-1] == action["polygon"]:
                self.polygons.pop()

        elif action["action"] == "cancel_current":
            self.current_polygon = action["polygon"]

        self.update()


class PolygonAnnotatorWindow(QWidget):
    """
    Main window: top area is the PolygonCanvas, bottombar contains Previous/Next buttons.
    """

    def __init__(self, image_paths, loop):
        """Build the AOI annotator window.

        Args:
            image_paths: Ordered list of image file paths to annotate.
            loop: QEventLoop to quit when the user clicks Finish.
        """
        super().__init__()
        self.close_flag = False
        self.loop = loop
        self.image_paths = image_paths
        self.index = 0
        self.results = {p: [] for p in image_paths}  # polygons in image coords

        self.canvas = PolygonCanvas(self)

        self.prev_btn = QPushButton("◀ Previous")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(lambda: self.change_image(-1))
        self.next_btn.clicked.connect(lambda: self.change_image(1))

        nav_btn_layout = QHBoxLayout()
        nav_btn_layout.addWidget(self.prev_btn, stretch=1)
        nav_btn_layout.addWidget(self.next_btn, stretch=1)

        self.finish_btn = QPushButton("Finish")
        self.finish_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14pt;
            }
            QPushButton:hover { background-color: #005a9e; }
            QPushButton:pressed { background-color: #004578; }
        """)
        self.finish_btn.clicked.connect(self.finish_annotation)

        finish_layout = QHBoxLayout()
        finish_layout.addWidget(self.finish_btn, stretch=1)

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

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addLayout(slider_layout)
        main_layout.addLayout(status_layout)
        main_layout.addLayout(nav_btn_layout)
        main_layout.addLayout(finish_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("AOI Annotator (ctrl+Z = undo, ctrl+Y = redo)")
        self.showMaximized()  # show first
        self.raise_()
        self.activateWindow()

        self.load_current_image()

    def finish_annotation(self):
        self.save_current_polygon()
        self.loop.quit()
        self.close_flag = True
        self.close()  # triggers closeEvent

    def update_line_thickness(self, value):
        """Propagate the slider value to the canvas and trigger a repaint."""
        self.canvas.line_thickness = value
        self.canvas.update()

    def closeEvent(self, event):
        """Exit the program if the user closed with X; accept normally after Finish."""
        if getattr(self, "close_flag", False):
            event.accept()
        else:
            print("Window closed — exiting program.")
            QApplication.quit()
            sys.exit(0)

    def load_current_image(self):
        """Load the image at the current index into the canvas, restoring any saved polygons."""
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
        """Update the "Image N / M" counter in the status bar."""
        self.status_label.setText(f"Image {self.index + 1} / {len(self.image_paths)}")

    def update_button_states(self):
        """Enable or disable Previous / Next based on the current index."""
        self.prev_btn.setEnabled(self.index > 0)
        self.next_btn.setEnabled(self.index < len(self.image_paths) - 1)

    def save_current_polygon(self):
        """Persist the canvas polygons for the current image into self.results."""
        path = self.image_paths[self.index]
        self.results[path] = self.canvas.get_polygon_image_coords()

    def change_image(self, delta):
        """Save the current polygons then navigate to the adjacent image by delta."""
        self.save_current_polygon()
        self.index = max(0, min(len(self.image_paths) - 1, self.index + delta))
        self.load_current_image()

    def keyPressEvent(self, event: QKeyEvent):
        """Ctrl+Z/Y for undo/redo, arrow keys to navigate images, Escape to close."""
        k = event.key()
        ctrl = event.modifiers() & Qt.ControlModifier

        if ctrl and k == Qt.Key_Z:
            self.canvas.undo()
            self.save_current_polygon()  # save immediately
        elif ctrl and k == Qt.Key_Y:
            self.canvas.redo()
            self.save_current_polygon()
        elif k == Qt.Key_Right:
            self.change_image(1)
        elif k == Qt.Key_Left:
            self.change_image(-1)
        elif k == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


def annotate_images(image_paths):
    """Launches GUI and returns {path: [[x,y], ...]} and line thickness"""
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    loop = QEventLoop()  # local loop to block until Finish/close

    win = PolygonAnnotatorWindow(image_paths, loop)
    win.show()

    loop.exec_()

    return win.results, win.canvas.line_thickness


# Example usage
if __name__ == "__main__":
    folder = "path/to/images"
    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    image_files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(valid_exts)
    ]

    polygons = annotate_images(image_files)
    for path, poly in polygons.items():
        print(path, poly)
