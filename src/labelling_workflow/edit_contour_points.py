# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import utils as utils
from PyQt5.QtGui import QPolygonF
from PyQt5.QtCore import QPointF
from utils import show_error_window

class ContourEditorView(QtWidgets.QGraphicsView):
    def __init__(self, image, contours, line_thickness, current_image_path=None, aois_dict=None, detected_contours=None, parent=None):
        super().__init__(parent)
        self.image = image
        self.selected_points = set()
        self.scaling_active = False
        self.mouse_pos = None
        self.scaling_initial_distance = None
        self.contours_original_for_scaling = None
        self.moving_active = False
        self.move_start_mouse_pos = None
        self.contours_original_for_move = None
        self.scaling_reference = {}
        self.contour_scale_factors = {}
        self.rotating_active = False
        self.rotation_reference = {}
        self.rotation_start_mouse_pos = None
        self.contours_original_for_rotation = None 
        self.creating_contour = False
        self.new_contour_points = [] 
        self.currently_creating_contour = False
        self.current_mode = "Selection"
        self.line_thickness = line_thickness
        self.current_image_path = current_image_path
        self.aois_dict = aois_dict if aois_dict is not None else {}
        
        if detected_contours is not None and len(contours) == 0:
            contours = [c.copy() for c in detected_contours]
            
        self.original_contours = contours
        self.contours = [c.copy() for c in contours]

        # Scene setup
        self.scene = QtWidgets.QGraphicsScene(self)
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(-10000, -10000, 20000, 20000)
        self.setScene(self.scene)

        # Rendering and interaction
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.zoom = 0.001
        self.pan_active = False
        self.last_mouse_pos = None
        self.undo_stack = []

        # Drawing
        self.drawing = False
        self.rect_item = None
        self.start_point = None
        self.initial_fit_done = False
        self.horizontalScrollBar().valueChanged.connect(self.viewport().update)
        self.verticalScrollBar().valueChanged.connect(self.viewport().update)

        # Initial render (no centering yet!)
        self.update_display()
        QtCore.QTimer.singleShot(0, self._initial_center_and_fit)
        
    def _initial_center_and_fit(self):
        """Center and fit the pixmap after the view has its final size."""
        if not self.pixmap_item.pixmap() or self.initial_fit_done:
            return

        self.resetTransform()
        # Fit the pixmap to the actual viewport size, keeping aspect ratio
        self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)
        # Center exactly on the pixmap
        self.centerOn(self.pixmap_item)
        self.initial_fit_done = True
        
    def clamp_to_image(self, x, y):
        """Clamp coordinates to stay within the image."""
        h, w = self.image.shape[:2]
        x_c = int(min(max(0, round(x)), w - 1))
        y_c = int(min(max(0, round(y)), h - 1))
        return x_c, y_c

    def showEvent(self, event):
        """Center and fit the image after the view is fully laid out."""
        super().showEvent(event)
        if not self.initial_fit_done:
            self.resetTransform()
            self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)
            self.centerOn(self.pixmap_item)
            self.initial_fit_done = True

    def update_display(self):
        """Draw the image and all contours."""
        img = self.image.copy()
        # Draw contours
        for c_idx, cnt in enumerate(self.contours):
            points = [tuple(pt[0]) for pt in cnt]
            color = (0, 0, 255)
            is_closed = not (self.creating_contour and c_idx == len(self.contours) - 1)
            if len(points) > 1:
                cv2.polylines(img, [np.array(points)], isClosed=is_closed, color=color, thickness=self.line_thickness)
            for pt_idx, point in enumerate(points):
                if (c_idx, pt_idx) in self.selected_points:
                    cv2.circle(img, point, self.line_thickness, (255, 0, 0), -1)
                else:
                    cv2.circle(img, point, int(self.line_thickness / 2), (0, 255, 0), -1)

        # Scaling/rotation guide
        if (self.scaling_active or self.rotating_active) and self.mouse_pos:
            img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
            x1, y1 = int(self.mouse_pos.x()), int(self.mouse_pos.y())
            x2, y2 = int(img_center.x()), int(img_center.y())
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), self.line_thickness)

        height, width, _ = img.shape
        qimage = QtGui.QImage(img.data, width, height, 3 * width, QtGui.QImage.Format_BGR888)
        self.pixmap_item.setPixmap(QtGui.QPixmap.fromImage(qimage))

        # No initial fit in update_display – let showEvent handle it
        # Extend scene rect for unlimited panning
        scene_rect = self.pixmap_item.boundingRect()
        scene_rect = scene_rect.adjusted(-10000, -10000, 10000, 10000)
        self.scene.setSceneRect(scene_rect)
        
    def paintEvent(self, event):
        super().paintEvent(event)  # Draw scene normally

        if self.current_mode:
            painter = QtGui.QPainter(self.viewport())
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0)))  # Red text
            font = QtGui.QFont("Arial", 20)
            font.setBold(True)
            painter.setFont(font)

            painter.drawText(50, 50, f"Mode: {self.current_mode}")

            painter.end()
            
    def keyPressEvent(self, event):
        
        if event.key() == QtCore.Qt.Key_U:
            if not self.creating_contour and not self.scaling_active and not self.moving_active and not self.rotating_active:
            
                if self.undo_stack:
                    self.contours = self.undo_stack.pop()
                    self.selected_points.clear()
                    self.update_display()

        elif event.key() == QtCore.Qt.Key_D:
            if not self.creating_contour and not self.scaling_active and not self.moving_active and not self.rotating_active:
                self.delete_selected_points()
                self.update_display()
                
        elif event.key() == QtCore.Qt.Key_Escape:
            sys.exit(0)  # quit python script entirely

        elif event.key() == QtCore.Qt.Key_S:
            if not self.creating_contour and not self.moving_active and not self.rotating_active and len(self.selected_points) > 0:
                
                self.scaling_active = not self.scaling_active
                if self.scaling_active:
                        self.current_mode = "Scaling" if self.scaling_active else ""
                        self.viewport().update()
                        self.undo_stack.append([c.copy() for c in self.contours])
                        cursor_pos = QtGui.QCursor.pos()
                        local_pos = self.mapFromGlobal(cursor_pos)
                        self.mouse_pos = self.mapToScene(local_pos)

                        # Save reference positions of selected points
                        self.scaling_reference = {}
                        self.contour_scale_factors = {}
                        self.contours_original_for_scaling = [c.copy() for c in self.contours]
                        
                        for c_idx, cnt in enumerate(self.contours):
                            any_selected = False
                            for pt_idx, pt in enumerate(cnt):
                                if tuple((c_idx, pt_idx)) in self.selected_points:
                                    self.scaling_reference[(c_idx, pt_idx)] = tuple(pt[0])
                                    any_selected = True
                            if any_selected:
                                self.contour_scale_factors[c_idx] = 1.0

                        # Record initial distance from center
                        img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
                        mouse_vec = np.array([
                            self.mouse_pos.x() - img_center.x(),
                            self.mouse_pos.y() - img_center.y()
                        ])
                        self.scaling_initial_distance = np.linalg.norm(mouse_vec)

                else:
                    self.current_mode = "Selection"
                    self.viewport().update()
                    self.mouse_pos = None
                    self.scaling_reference = {}
                    self.contour_scale_factors = {}

                self.update_display()
            
        elif event.key() == QtCore.Qt.Key_M:
            if not self.creating_contour and not self.scaling_active and not self.rotating_active and len(self.selected_points) > 0:
                self.moving_active = not self.moving_active
                if self.moving_active:
                    self.current_mode = "Moving" if self.moving_active else ""
                    self.viewport().update()
                    self.undo_stack.append([c.copy() for c in self.contours])
                    cursor_pos = QtGui.QCursor.pos()
                    local_pos = self.mapFromGlobal(cursor_pos)
                    self.move_start_mouse_pos = self.mapToScene(local_pos)
                    self.contours_original_for_move = [c.copy() for c in self.contours]
                else:
                    self.current_mode = "Selection"
                    self.viewport().update()
                    self.move_start_mouse_pos = None
                    self.contours_original_for_move = None
                self.update_display()
            
        elif event.key() == QtCore.Qt.Key_R:
            if not self.creating_contour and not self.scaling_active and not self.moving_active and len(self.selected_points) > 0:
                self.rotating_active = not self.rotating_active
                if self.rotating_active:
                    self.current_mode = "Rotating" if self.rotating_active else ""
                    self.viewport().update()
                    self.undo_stack.append([c.copy() for c in self.contours])
                    cursor_pos = QtGui.QCursor.pos()
                    local_pos = self.mapFromGlobal(cursor_pos)
                    self.mouse_pos = self.mapToScene(local_pos)
                    self.rotation_reference = {}
                    self.contours_original_for_rotation = [c.copy() for c in self.contours]

                    img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
                    start_vec = np.array([
                        self.mouse_pos.x() - img_center.x(),
                        self.mouse_pos.y() - img_center.y()
                    ])
                    self.rotation_start_angle = np.arctan2(start_vec[1], start_vec[0])

                    for c_idx, cnt in enumerate(self.contours):
                        for pt_idx, pt in enumerate(cnt):
                            if (c_idx, pt_idx) in self.selected_points:
                                self.rotation_reference[(c_idx, pt_idx)] = tuple(pt[0])
                else:
                    self.current_mode = "Selection"
                    self.viewport().update()
                    self.mouse_pos = None
                    self.rotation_reference = {}
                    self.contours_original_for_rotation = None
                    self.rotation_start_angle = None
                self.update_display()
            
        elif event.key() == QtCore.Qt.Key_C:
            if self.moving_active or self.scaling_active or self.rotating_active:
                return  # Don't allow drawing during other modes

            if self.creating_contour:
                # Finish and close contour
                self.current_mode = "Selection"
                self.viewport().update()
                self.currently_creating_contour = False
                if len(self.new_contour_points) < 3:
                    self.contours.pop()  
                self.new_contour_points.clear()
                self.creating_contour = False
            else:
                # Begin new contour
                self.current_mode = "Contour Creation" if not self.creating_contour else ""
                self.viewport().update()
                self.undo_stack.append([c.copy() for c in self.contours])
                self.creating_contour = True
                self.new_contour_points.clear()
                self.contours.append(np.array([], dtype=np.int32).reshape((-1, 1, 2)))

            self.update_display()

        else:
            super().keyPressEvent(event)
        
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # Deactivate all modes if not creating a contour
            if not self.creating_contour:
                self.scaling_active = False
                self.moving_active = False
                self.rotating_active = False
                self.mouse_pos = None
                self.scaling_reference = {}
                self.rotation_reference = {}
                self.contours_original_for_move = None
                self.contours_original_for_rotation = None
                self.rotation_start_angle = None
                self.currently_creating_contour = False
                self.current_mode = "Selection"
                self.viewport().update()

            # Start lasso drawing
            if not self.creating_contour and not self.scaling_active and not self.moving_active and not self.rotating_active:
                self.drawing = True
                self.lasso_points = [self.mapToScene(event.pos())]

            if self.rect_item:
                self.scene.removeItem(self.rect_item)
                self.rect_item = None
            if hasattr(self, 'lasso_item') and self.lasso_item:
                self.scene.removeItem(self.lasso_item)
                self.lasso_item = None

            # --- Handle contour creation ---
            if self.creating_contour:
                if not self.currently_creating_contour:
                    self.currently_creating_contour = True

                scene_pos = self.mapToScene(event.pos())
                x, y = int(scene_pos.x()), int(scene_pos.y())

                # --- AOI check ---
                current_aoi = self.aois_dict.get(self.current_image_path, [])
                if current_aoi and len(current_aoi) >= 3:
                    aoi_polygon = QPolygonF([QPointF(px, py) for px, py in current_aoi])
                else:
                    aoi_polygon = None

                # Only add point if inside AOI or no AOI defined
                if aoi_polygon is None or aoi_polygon.containsPoint(QPointF(x, y), QtCore.Qt.OddEvenFill):
                    x, y = self.clamp_to_image(x, y)
                    self.new_contour_points.append((x, y))
                    self.contours[-1] = np.array(self.new_contour_points, dtype=np.int32).reshape((-1, 1, 2))
                else:
                    print("Contour point outside AOI; ignoring")

            self.update_display()

        elif event.button() == QtCore.Qt.RightButton:
            self.pan_active = True
            self.last_mouse_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

        super().mousePressEvent(event)
        
    def move_selected_points(self):
        if not self.selected_points or not self.move_start_mouse_pos or not self.mouse_pos:
            return

        # Calculate movement delta from move start
        dx = self.mouse_pos.x() - self.move_start_mouse_pos.x()
        dy = self.mouse_pos.y() - self.move_start_mouse_pos.y()

        # Reset contours to original snapshot to avoid cumulative translation
        self.contours = [c.copy() for c in self.contours_original_for_move]

        for c_idx, cnt in enumerate(self.contours):
            for pt_idx, pt in enumerate(cnt):
                if (c_idx, pt_idx) in self.selected_points:
                    x, y = pt[0]
                    new_x, new_y = self.clamp_to_image(x + dx, y + dy)
                    cnt[pt_idx][0] = [new_x, new_y]

    def mouseMoveEvent(self, event):
        self.mouse_pos = self.mapToScene(event.pos())
        if self.drawing:
            point = self.mouse_pos
            self.lasso_points.append(point)
            if hasattr(self, 'lasso_item') and self.lasso_item:
                self.scene.removeItem(self.lasso_item)
            polygon = QtGui.QPolygonF(self.lasso_points)
            pen = QtGui.QPen(QtGui.QColor("red"))
            pen.setWidth(self.line_thickness)  # use the slider value
            self.lasso_item = self.scene.addPolygon(polygon, pen)
        elif self.pan_active:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.viewport().update()
        elif self.scaling_active:
            self.scale_selected_points()
        elif self.moving_active:
            self.move_selected_points()
        elif self.rotating_active:
            self.rotate_selected_points()
        self.update_display()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == QtCore.Qt.LeftButton:
            self.drawing = False

            # If fewer than 3 points, treat it as a click: deselect all
            if hasattr(self, 'lasso_points') and len(self.lasso_points) < 3:
                self.selected_points.clear()

            elif hasattr(self, 'lasso_points') and len(self.lasso_points) >= 3:
                polygon = QtGui.QPolygonF(self.lasso_points)
                self.select_points_in_polygon(polygon)

            if hasattr(self, 'lasso_item') and self.lasso_item:
                self.scene.removeItem(self.lasso_item)
                self.lasso_item = None

            self.update_display()

        elif event.button() == QtCore.Qt.RightButton:
            self.pan_active = False
            self.setCursor(QtCore.Qt.ArrowCursor)

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Zoom on scroll
        delta = event.angleDelta().y()
        zoom_factor = 1.2 if delta > 0 else 1 / 1.2
        self.zoom *= zoom_factor
        self.scale(zoom_factor, zoom_factor)

    def select_points_in_polygon(self, polygon):
        self.selected_points.clear()
        for c_idx, cnt in enumerate(self.contours):
            for pt_idx, pt in enumerate(cnt):
                x, y = pt[0]
                pointf = QtCore.QPointF(x, y)
                if polygon.containsPoint(pointf, QtCore.Qt.OddEvenFill):
                    self.selected_points.add((c_idx, pt_idx))

    def get_edited_contours(self):
        return self.contours
    
    def delete_selected_points(self):
        new_contours = []
        for c_idx, cnt in enumerate(self.contours):
            new_cnt = []
            for pt_idx, pt in enumerate(cnt):
                if (c_idx, pt_idx) not in self.selected_points:
                    new_cnt.append(pt)
            if new_cnt:
                new_contours.append(np.array(new_cnt, dtype=np.int32))
        self.undo_stack.append(self.contours.copy())
        self.contours = new_contours
        self.selected_points.clear()

    def rotate_selected_points(self):
        if not self.rotation_reference or not self.mouse_pos or self.rotation_start_angle is None:
            return

        img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
        current_vec = np.array([
            self.mouse_pos.x() - img_center.x(),
            self.mouse_pos.y() - img_center.y()
        ])
        current_angle = np.arctan2(current_vec[1], current_vec[0])

        angle_delta = current_angle - self.rotation_start_angle

        for (c_idx, pt_idx), orig_pt in self.rotation_reference.items():
            cnt = self.contours_original_for_rotation[c_idx]

            # Compute centroid of the contour
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            ox, oy = orig_pt
            dx, dy = ox - cx, oy - cy
            radius = np.sqrt(dx**2 + dy**2)
            original_angle = np.arctan2(dy, dx)
            new_angle = original_angle + angle_delta

            new_x = cx + radius * np.cos(new_angle)
            new_y = cy + radius * np.sin(new_angle)
            new_x, new_y = self.clamp_to_image(new_x, new_y)

            self.contours[c_idx][pt_idx][0] = [int(new_x), int(new_y)]
        
    def scale_selected_points(self):
        if not self.selected_points or not self.mouse_pos or not self.scaling_initial_distance:
            return

        img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
        mouse_vec = np.array([
            self.mouse_pos.x() - img_center.x(),
            self.mouse_pos.y() - img_center.y()
        ])
        current_distance = np.linalg.norm(mouse_vec)

        # Relative scaling factor from initial mouse distance
        new_scale = current_distance / self.scaling_initial_distance

        updated_contours = set()

        for (c_idx, pt_idx), orig_pt in self.scaling_reference.items():
            cnt_orig = self.contours_original_for_scaling[c_idx]
            cnt = self.contours[c_idx]

            # Compute contour centroid from original
            M = cv2.moments(cnt_orig)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            ox, oy = cnt_orig[pt_idx][0]
            dx, dy = ox - cx, oy - cy
            new_x = int(cx + dx * new_scale)
            new_y = int(cy + dy * new_scale)
            new_x, new_y = self.clamp_to_image(new_x, new_y)
            cnt[pt_idx][0] = [new_x, new_y]

            updated_contours.add(c_idx)

        for c_idx in updated_contours:
            self.contour_scale_factors[c_idx] = new_scale
            
class MultiImageContourEditor(QtWidgets.QWidget):
    def __init__(self, image_dict, line_thickness=2, detected_contours=None):
        """
        image_dict: dict[str, np.ndarray]
            Keys are image identifiers (usually original paths),
            values are numpy images already loaded in memory.
        """
        super().__init__()
        self.image_dict = image_dict
        self.image_keys = list(image_dict.keys())
        self.line_thickness = line_thickness
        self.index = 0
        self.results = {k: [] for k in self.image_keys}  # stores contours per image
        self.detected_contours = detected_contours or {}

        # Layouts
        self.layout = QtWidgets.QVBoxLayout(self)
        self.button_layout = QtWidgets.QHBoxLayout()

        # Navigation buttons
        self.prev_btn = QtWidgets.QPushButton("◀ Previous")
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.prev_btn.clicked.connect(lambda: self.change_image(-1))
        self.next_btn.clicked.connect(lambda: self.change_image(1))
        self.button_layout.addWidget(self.prev_btn)
        self.button_layout.addWidget(self.next_btn)

        # Status label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)

        # Add button layout and status label to main layout
        self.layout.addWidget(self.status_label)
        self.layout.addLayout(self.button_layout)

        # Initialize ContourEditorView (empty for now)
        self.editor_view = None

        # Load first image
        self.load_image(self.index)

        self.showMaximized()

        def focus_window():
            self.raise_()                 # bring window to top
            self.activateWindow()         # make it the active window
            self.setFocus(QtCore.Qt.ActiveWindowFocusReason)
        # Force reload after window has fully shown
        QtCore.QTimer.singleShot(0, focus_window)
        
        QtCore.QTimer.singleShot(0, lambda: self.change_image(0, force=True))

                
    def update_navigation_buttons(self):
        self.prev_btn.setEnabled(self.index > 0)
        self.next_btn.setEnabled(self.index < len(self.image_keys) - 1)

    def load_image(self, idx):
        """Load image at index `idx` into ContourEditorView."""
        key = self.image_keys[idx]
        img = self.image_dict[key]
        if img is None or not isinstance(img, np.ndarray):
            show_error_window(f"Image at key '{key}' is not a valid numpy array.")
            return

        # Save current contours before switching
        if self.editor_view is not None:
            self.results[self.image_keys[self.index]] = self.editor_view.get_edited_contours()
            self.layout.removeWidget(self.editor_view)
            self.editor_view.deleteLater()

        # Get saved contours (from previous editing or from detection)
        saved_contours = self.results.get(key, [])

        # Load detected contours if available
        detected_contours = self.detected_contours.get(key, None)
        if detected_contours is not None and len(saved_contours) == 0:
            # Use a copy so original dict is not modified
            detected_contours = [c.copy() for c in detected_contours]
        else:
            detected_contours = None

        self.editor_view = ContourEditorView(
            img.copy(),
            contours=[c.copy() for c in saved_contours] if saved_contours else [],
            line_thickness=self.line_thickness,
            detected_contours=detected_contours
        )

        # Add editor view to layout
        self.layout.insertWidget(0, self.editor_view, stretch=1)
        self.index = idx
        self.update_status()
        self.update_navigation_buttons()

        # Force focus so key events work immediately
        QtCore.QTimer.singleShot(0, lambda: self.editor_view.viewport().setFocus(QtCore.Qt.ActiveWindowFocusReason))

        # Fix centering: defer until layout done
        def center_when_ready():
            if self.editor_view.viewport().width() > 0 and self.editor_view.viewport().height() > 0:
                self.editor_view._initial_center_and_fit()
            else:
                QtCore.QTimer.singleShot(10, center_when_ready)

        QtCore.QTimer.singleShot(0, center_when_ready)

        
    def change_image(self, delta, force=False):
        new_index = max(0, min(len(self.image_keys) - 1, self.index + delta))
        if force or new_index != self.index:
            self.load_image(new_index)

    def update_status(self):
        self.status_label.setText(f"Image {self.index + 1} / {len(self.image_keys)}")

    def get_results(self):
        # Save the currently edited image
        if self.editor_view is not None:
            self.results[self.image_keys[self.index]] = self.editor_view.get_edited_contours()
        return self.results
        

def run_contour_editor(image_dict, line_thickness=2, detected_contours=None):
    """
    Launch a contour editor for one or more in-memory images.

    Args:
        image_dict (dict): {image_path: np.ndarray}.
        line_thickness (int): Thickness of drawn lines/points.

    Returns:
        dict: {image_path: contours}, where contours are lists of numpy arrays.
    """
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        created_app = True

    editor_widget = MultiImageContourEditor(image_dict, line_thickness=line_thickness, detected_contours=detected_contours)
    editor_widget.setWindowTitle("Multi-Image Contour Editor (C=Create Contour, D=Delete, U=Undo, S=Scale, M=Move, R=Rotate)")
    editor_widget.setWindowFlags(editor_widget.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
    editor_widget.showMaximized()
    editor_widget.setWindowFlags(editor_widget.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
    editor_widget.showMaximized()

    if created_app:
        app.exec_()
    else:
        app.exec_()

    return editor_widget.get_results()
