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
        self.undo_stack = []
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
        self.orig_image = image_bgr
        self.orig_h, self.orig_w = image_bgr.shape[:2]
        self.update_scaled_image()

        # FULL reset per image
        self.polygons = []
        self.current_polygon = {"points": [], "closed": False}
        self.undo_stack = []
        self.hover_first = False

        # Restore saved polygons (image → widget coords)
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

        # convert to QImage/QPixmap
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
        # scale with keep aspect ratio
        scaled = pix.scaled(w_win, h_win, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scaled_pixmap = scaled

        # offsets to center
        self.offset_x = (w_win - scaled.width()) // 2
        self.offset_y = (h_win - scaled.height()) // 2

        # uniform scale from image -> widget
        self.scale = scaled.width() / max(1, self.orig_w)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        pts = self.current_polygon["points"]

        # Handle hover near first point
        if len(pts) > 0 and not self.current_polygon["closed"]:
            first_pt = pts[0]
            dist_sq = (first_pt.x() - pos.x()) ** 2 + (first_pt.y() - pos.y()) ** 2
            self.hover_first = dist_sq <= 100
        else:
            if self.hover_first:
                self.hover_first = False

        # If dragging, add points as we move
        if self.dragging:
            if (
                self.last_drag_point is None
                or (pos - self.last_drag_point).manhattanLength() > 15
            ):
                pts.append(pos)
                self.last_drag_point = pos

        self.update()

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
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        if self.scaled_pixmap:
            painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)

        # --- Draw completed polygons ---
        for poly in self.polygons:
            pts = poly["points"]

            # draw lines in green
            pen_lines = QPen(QColor(0, 200, 0), self.line_thickness)
            painter.setPen(pen_lines)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            if poly["closed"] and len(pts) > 2:
                painter.drawLine(pts[-1], pts[0])

            # draw points in red
            pen_points = QPen(QColor(200, 0, 0))
            painter.setPen(pen_points)
            painter.setBrush(QColor(200, 0, 0))
            for pt in pts:
                painter.drawEllipse(pt, 5, 5)

        # --- Draw current polygon ---
        pts = self.current_polygon["points"]

        # draw lines in green
        pen_lines = QPen(QColor(0, 200, 0), self.line_thickness)
        painter.setPen(pen_lines)
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])

        # draw points
        for i, pt in enumerate(pts):
            if i == 0 and self.hover_first:
                color = QColor(0, 0, 255)  # first point hovered -> blue
            else:
                color = QColor(200, 0, 0)
            painter.setPen(QPen(color))
            painter.setBrush(color)
            painter.drawEllipse(pt, 5, 5)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or not self.scaled_pixmap:
            return

        pos = event.pos()
        self.dragging = False
        self.last_drag_point = None

        pts = self.current_polygon["points"]
        if not pts:
            return

        # If released near first point, close polygon
        if len(pts) >= 3:
            first_pt = pts[0]
            dist_sq = (first_pt.x() - pos.x()) ** 2 + (first_pt.y() - pos.y()) ** 2
            if len(pts) >= 3 and dist_sq <= 100:
                self.close_polygon()
                return
            else:
                # Add final point on release
                pts.append(pos)

        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or not self.scaled_pixmap:
            return

        pos = event.pos()

        # Only accept clicks inside the image
        if not (
            self.offset_x <= pos.x() <= self.offset_x + self.scaled_pixmap.width()
            and self.offset_y <= pos.y() <= self.offset_y + self.scaled_pixmap.height()
        ):
            return

        pts = self.current_polygon["points"]
        self.dragging = True
        self.last_drag_point = pos

        # If first point clicked (and polygon has ≥3 points), close polygon
        if len(pts) >= 3:
            first_pt = pts[0]
            dist_sq = (first_pt.x() - pos.x()) ** 2 + (first_pt.y() - pos.y()) ** 2
            if dist_sq <= 100:  # within 10px
                self.close_polygon()
                return

        # Otherwise, add point to current polygon
        pts.append(pos)
        self.update()

    def resizeEvent(self, event):
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

    # --- utility to get polygon in image coords ---
    def get_polygon_image_coords(self):
        all_polys = []
        for poly in self.polygons:
            pts_img = [self.widget_to_image_coords(pt) for pt in poly["points"]]
            all_polys.append(pts_img)
        return all_polys

    def close_polygon(self):
        """Close the current polygon and store it in polygons list."""
        if len(self.current_polygon["points"]) > 2:
            self.current_polygon["closed"] = True
            self.polygons.append(self.current_polygon)
            self.undo_stack.append({"action": "add", "polygon": self.current_polygon})
            self.current_polygon = {
                "points": [],
                "closed": False,
            }  # ready for next polygon
            self.update()
            return True
        return False

    def undo(self):
        """Undo last action: either cancel current polygon or remove last completed polygon."""
        if self.current_polygon["points"]:
            # cancel in-progress polygon
            self.undo_stack.append(
                {"action": "cancel_current", "polygon": self.current_polygon}
            )
            self.current_polygon = {"points": [], "closed": False}
            self.hover_first = False
        elif self.polygons:
            # remove last completed polygon
            last = self.polygons.pop()
            self.undo_stack.append({"action": "remove", "polygon": last})
        self.update()


class PolygonAnnotatorWindow(QWidget):
    """
    Main window: top area is the PolygonCanvas, bottombar contains Previous/Next buttons.
    """

    def __init__(self, image_paths, loop):
        super().__init__()
        self.close_flag = False
        self.loop = loop
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
        nav_btn_layout = QHBoxLayout()
        nav_btn_layout.addWidget(self.prev_btn, stretch=1)
        nav_btn_layout.addWidget(self.next_btn, stretch=1)

        # Bottom row: Finish button (blue)
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
        main_layout.addLayout(nav_btn_layout)
        main_layout.addLayout(finish_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("AOI Annotator (ctrl+Z = undo)")
        self.showMaximized()
        self.showMaximized()  # show first
        self.raise_()
        self.activateWindow()

        # Load first image
        self.load_current_image()

    def finish_annotation(self):
        """Called when Finish button is clicked"""
        self.save_current_polygon()
        self.loop.quit()  # stop local loop
        self.close_flag = True
        self.close()  # triggers closeEvent

    def update_line_thickness(self, value):
        self.canvas.line_thickness = value
        self.canvas.update()

    def closeEvent(self, event):
        """Called when the window is closed"""
        if getattr(self, "close_flag", False):
            # Finish button triggered — just close window, do not exit program
            event.accept()
        else:
            # User clicked X — fully exit program
            print("Window closed — exiting program.")
            QApplication.quit()
            sys.exit(0)

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
        ctrl = event.modifiers() & Qt.ControlModifier

        if ctrl and k == Qt.Key_Z:
            self.canvas.undo()
            self.save_current_polygon()  # save immediately
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

    # block here until user clicks Finish or closes the window
    loop.exec_()

    # return polygons and final line thickness
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
