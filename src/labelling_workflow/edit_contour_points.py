# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import utils as utils
from PyQt5.QtGui import QPolygonF
from PyQt5.QtCore import QPointF, Qt


class ContourEditorView(QtWidgets.QGraphicsView):
    def __init__(
        self,
        image,
        contours,
        line_thickness,
        current_image_path=None,
        aois_dict=None,
        detected_contours=None,
        parent=None,
    ):
        super().__init__(parent)
        self.image = image
        self.selected_points = set()
        self.scaling_active = False
        self.redo_stack = []
        self.contour_creation_undo_stack = None
        self.mouse_pos = None
        self.scaling_initial_distance = None
        self.contours_original_for_scaling = None
        self.contour_creation_redo_stack = None
        self.moving_active = False
        self.move_start_mouse_pos = None
        self.contours_original_for_move = None
        self.scaling_reference = {}
        self.contour_scale_factors = {}
        self.rotating_active = False
        self.freehand_drawing = False
        self.last_freehand_point = None
        self.freehand_min_dist = 10
        self.rotation_reference = {}
        self.rotation_start_mouse_pos = None
        self.contours_original_for_rotation = None
        self.creating_contour = False
        self.hovering_start_point = False
        self.start_point_radius = 10
        self.new_contour_points = []
        self.currently_creating_contour = False
        self.contour_items = []  # QGraphicsPathItem
        self.point_items = []
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
        # self.scene.setSceneRect(-10000, -10000, 20000, 20000)
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

    def _clear_redo(self):
        self.redo_stack.clear()

    def _initial_center_and_fit(self):
        """Center the pixmap in the view and scale to fit the window."""
        if not self.pixmap_item.pixmap():
            return

        self.resetTransform()
        self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)
        self.centerOn(self.pixmap_item)

        # Optional: expand scene for panning far around the image
        padding = 5000  # can pan 5000 px around
        pixmap_rect = self.pixmap_item.boundingRect()
        big_rect = pixmap_rect.adjusted(-padding, -padding, padding, padding)
        self.scene.setSceneRect(big_rect)

        self.initial_fit_done = True

    def clamp_to_image(self, x, y):
        """Clamp coordinates to stay within the image."""
        h, w = self.image.shape[:2]
        x_c = int(min(max(0, round(x)), w - 1))
        y_c = int(min(max(0, round(y)), h - 1))
        return x_c, y_c

    def _rebuild_contour_items(self):
        # Remove old items
        for item in self.contour_items + self.point_items:
            self.scene.removeItem(item)

        self.contour_items.clear()
        self.point_items.clear()

        for c_idx, cnt in enumerate(self.contours):
            if len(cnt) == 0:
                continue

            # ---- build contour path ----
            path = QtGui.QPainterPath()
            pts = [QtCore.QPointF(x, y) for x, y in cnt.reshape(-1, 2)]
            path.moveTo(pts[0])
            for p in pts[1:]:
                path.lineTo(p)

            if not (self.creating_contour and c_idx == len(self.contours) - 1):
                path.closeSubpath()

            pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
            pen.setWidth(self.line_thickness)
            pen.setCosmetic(True)

            path_item = QtWidgets.QGraphicsPathItem(path)
            path_item.setPen(pen)
            path_item.setZValue(10)

            self.scene.addItem(path_item)
            self.contour_items.append(path_item)

            # ---- build points ----
            for pt_idx, (x, y) in enumerate(cnt.reshape(-1, 2)):
                color = QtGui.QColor(0, 255, 0)  # green

                # hovering start point
                if (
                    self.creating_contour
                    and c_idx == len(self.contours) - 1
                    and pt_idx == 0
                    and self.hovering_start_point
                    and len(cnt) >= 3
                ):
                    color = QtGui.QColor(0, 0, 255)  # blue

                if (c_idx, pt_idx) in self.selected_points:
                    color = QtGui.QColor(255, 0, 0)  # red

                r = self.line_thickness * 2  # SCREEN pixels

                item = QtWidgets.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
                item.setPos(x, y)

                item.setBrush(QtGui.QBrush(color))
                item.setPen(QtGui.QPen(Qt.NoPen))

                # 🔥 THIS is the missing line
                item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)

                item.setZValue(20)

                self.scene.addItem(item)
                self.point_items.append(item)

    def update_display(self):
        h, w = self.image.shape[:2]
        qimage = QtGui.QImage(self.image.data, w, h, 3 * w, QtGui.QImage.Format_BGR888)
        self.pixmap_item.setPixmap(QtGui.QPixmap.fromImage(qimage))

        self._rebuild_contour_items()

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

        k = event.key()
        ctrl = event.modifiers() & Qt.ControlModifier

        if ctrl and k == QtCore.Qt.Key_Z:
            if self.creating_contour and self.contour_creation_undo_stack:
                current_stack = self.contour_creation_undo_stack[-1]

                if current_stack:
                    # Save current state for redo
                    self.contour_creation_redo_stack.append(
                        self.new_contour_points.copy()
                    )

                    # Restore previous state
                    self.new_contour_points = current_stack.pop()

                    if self.contours:
                        self.contours.pop()

                    if self.new_contour_points:
                        self.contours.append(
                            np.array(self.new_contour_points, dtype=np.int32).reshape(
                                (-1, 1, 2)
                            )
                        )
                    else:
                        self.contours.append(np.empty((0, 1, 2), dtype=np.int32))

                    self.update_display()

                else:
                    # Current stack is empty → remove it
                    self.contour_creation_undo_stack.pop()
                    if self.contour_creation_undo_stack[-1]:
                        self.contour_creation_undo_stack[-1].pop()
                    if self.contours:
                        self.contours.pop()
                    if self.contours:
                        self.new_contour_points = (
                            self.contours[-1].reshape(-1, 2).tolist()
                        )
                    self.contour_creation_redo_stack.append("close")
                    self.undo_stack.pop()
                    self.update_display()

                return

            # --- normal undo stack ---
            if (
                not self.creating_contour
                and not self.scaling_active
                and not self.moving_active
                and not self.rotating_active
            ):
                if self.undo_stack:
                    current = [c.copy() for c in self.contours]

                    while self.undo_stack:
                        prev = self.undo_stack.pop()

                        # save redo only once, for the first actual undo
                        self.redo_stack.append(current)

                        # compare contours by value
                        same = len(prev) == len(current) and all(
                            np.array_equal(a, b) for a, b in zip(prev, current)
                        )

                        self.contours = prev

                        if not same:
                            break

                    self.selected_points.clear()
                    self.update_display()

        elif ctrl and k == QtCore.Qt.Key_Y:
            if self.creating_contour and self.contour_creation_redo_stack:
                # Save current state back to undo
                self.contour_creation_undo_stack[-1].append(
                    self.new_contour_points.copy()
                )

                # Restore redo state
                redo_item = self.contour_creation_redo_stack.pop()

                if redo_item == "close":
                    # Close current polygon
                    if self.contours:
                        self.contours[-1] = np.array(
                            self.new_contour_points, dtype=np.int32
                        ).reshape((-1, 1, 2))
                    self.new_contour_points.clear()
                    self.currently_creating_contour = False
                    self.hovering_start_point = False

                    # Start a new empty contour
                    self.contour_creation_undo_stack.append([])
                    self.contours.append(np.empty((0, 1, 2), dtype=np.int32))
                    self.contour_creation_undo_stack.append([])

                else:
                    # Normal redo of points
                    self.new_contour_points = redo_item.copy()
                    if self.contours:
                        self.contours[-1] = np.array(
                            self.new_contour_points, dtype=np.int32
                        ).reshape((-1, 1, 2))
                    else:
                        self.contours.append(
                            np.array(self.new_contour_points, dtype=np.int32).reshape(
                                (-1, 1, 2)
                            )
                        )

                self.update_display()
                return

            if (
                not self.creating_contour
                and not self.scaling_active
                and not self.moving_active
                and not self.rotating_active
            ):
                if self.redo_stack:
                    # save current state back to undo
                    self.undo_stack.append([c.copy() for c in self.contours])

                    # restore redo state
                    self.contours = self.redo_stack.pop()
                    self.selected_points.clear()
                    self.update_display()

        elif k == QtCore.Qt.Key_U:
            if (
                not self.creating_contour
                and not self.scaling_active
                and not self.moving_active
                and not self.rotating_active
                and len({c for c, _ in self.selected_points}) >= 2
            ):
                self.union_selected_contours()
                self.remove_small_contours()
                self.update_display()

        elif k == QtCore.Qt.Key_D:
            if (
                not self.creating_contour
                and not self.scaling_active
                and not self.moving_active
                and not self.rotating_active
            ):
                self.delete_selected_points()
                self.remove_small_contours()
                self.update_display()

        elif k == QtCore.Qt.Key_Escape:

            if self.creating_contour:
                self.creating_contour = False
                self.currently_creating_contour = False
                self.new_contour_points.clear()
                self.contour_creation_undo_stack = None
                self.hovering_start_point = False

                # Remove empty trailing contour
                if self.contours and len(self.contours[-1]) == 0:
                    self.contours.pop()

                self.current_mode = "Selection"
                self.contour_creation_redo_stack = None
                self.viewport().update()
                self.remove_small_contours()
                self.update_display()
                return

            # 2. Cancel scaling
            if self.scaling_active:
                self.scaling_active = False
                self.mouse_pos = None
                self.scaling_reference = {}
                self.contour_scale_factors = {}
                self.contours = [c.copy() for c in self.contours_original_for_scaling]
                self.current_mode = "Selection"
                self.update_display()
                return

            # 3. Cancel rotation
            if self.rotating_active:
                self.rotating_active = False
                self.mouse_pos = None
                self.rotation_reference = {}
                self.contours = [c.copy() for c in self.contours_original_for_rotation]
                self.rotation_start_angle = None
                self.current_mode = "Selection"
                self.update_display()
                return

            # 4. Cancel moving
            if self.moving_active:
                self.moving_active = False
                self.move_start_mouse_pos = None
                self.contours = [c.copy() for c in self.contours_original_for_move]
                self.current_mode = "Selection"
                self.update_display()
                return

            # 5. Cancel lasso selection
            if self.drawing:
                self.drawing = False
                self.lasso_points = []
                if hasattr(self, "lasso_item") and self.lasso_item:
                    self.scene.removeItem(self.lasso_item)
                    self.lasso_item = None
                self.viewport().update()
                return

            return

        elif k == QtCore.Qt.Key_S:
            if (
                not self.creating_contour
                and not self.moving_active
                and not self.rotating_active
                and len(self.selected_points) > 0
            ):

                self.scaling_active = not self.scaling_active
                if self.scaling_active:
                    self.current_mode = "Scaling" if self.scaling_active else ""
                    self.viewport().update()
                    self.undo_stack.append([c.copy() for c in self.contours])
                    self._clear_redo()
                    cursor_pos = QtGui.QCursor.pos()
                    local_pos = self.mapFromGlobal(cursor_pos)
                    self.mouse_pos = self.mapToScene(local_pos)

                    # Save reference positions of selected points
                    self.scaling_reference = {}
                    self.contour_scale_factors = {}
                    self.contours_original_for_scaling = [
                        c.copy() for c in self.contours
                    ]

                    for c_idx, cnt in enumerate(self.contours):
                        any_selected = False
                        for pt_idx, pt in enumerate(cnt):
                            if tuple((c_idx, pt_idx)) in self.selected_points:
                                self.scaling_reference[(c_idx, pt_idx)] = tuple(pt[0])
                                any_selected = True
                        if any_selected:
                            self.contour_scale_factors[c_idx] = 1.0

                    # Record initial distance from center
                    img_center = QtCore.QPointF(
                        self.image.shape[1] / 2, self.image.shape[0] / 2
                    )
                    mouse_vec = np.array(
                        [
                            self.mouse_pos.x() - img_center.x(),
                            self.mouse_pos.y() - img_center.y(),
                        ]
                    )
                    self.scaling_initial_distance = np.linalg.norm(mouse_vec)

                else:
                    self.current_mode = "Selection"
                    self.viewport().update()
                    self.mouse_pos = None
                    self.scaling_reference = {}
                    self.contour_scale_factors = {}

                self.update_display()

        elif k == QtCore.Qt.Key_M:
            if (
                not self.creating_contour
                and not self.scaling_active
                and not self.rotating_active
                and len(self.selected_points) > 0
            ):
                self.moving_active = not self.moving_active
                if self.moving_active:
                    self.current_mode = "Moving" if self.moving_active else ""
                    self.viewport().update()
                    self.undo_stack.append([c.copy() for c in self.contours])
                    self._clear_redo()
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

        elif k == QtCore.Qt.Key_R:
            if (
                not self.creating_contour
                and not self.scaling_active
                and not self.moving_active
                and len(self.selected_points) > 0
            ):
                self.rotating_active = not self.rotating_active
                if self.rotating_active:
                    self.current_mode = "Rotating" if self.rotating_active else ""
                    self.viewport().update()
                    self.undo_stack.append([c.copy() for c in self.contours])
                    self._clear_redo()
                    cursor_pos = QtGui.QCursor.pos()
                    local_pos = self.mapFromGlobal(cursor_pos)
                    self.mouse_pos = self.mapToScene(local_pos)
                    self.rotation_reference = {}
                    self.contours_original_for_rotation = [
                        c.copy() for c in self.contours
                    ]

                    img_center = QtCore.QPointF(
                        self.image.shape[1] / 2, self.image.shape[0] / 2
                    )
                    start_vec = np.array(
                        [
                            self.mouse_pos.x() - img_center.x(),
                            self.mouse_pos.y() - img_center.y(),
                        ]
                    )
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

        elif k == QtCore.Qt.Key_C:
            if self.creating_contour:
                self.creating_contour = False
                self.currently_creating_contour = False
                self.new_contour_points.clear()
                self.contour_creation_undo_stack = None
                self.contour_creation_redo_stack = None

                # Remove empty trailing contour if it exists
                if self.contours and len(self.contours[-1]) == 0:
                    self.contours.pop()

                self.current_mode = "Selection"
                self.viewport().update()
                self.remove_small_contours()
                self.update_display()
            else:
                # ENTER contour creation
                self.current_mode = "Contour Creation"
                self.viewport().update()
                self.undo_stack.append([c.copy() for c in self.contours])
                self._clear_redo()

                self.creating_contour = True
                self.currently_creating_contour = False
                self.new_contour_points = []
                self.contour_creation_undo_stack = [[]]
                self.contour_creation_redo_stack = []

                self.contours.append(np.empty((0, 1, 2), dtype=np.int32))
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

            # Start lasso drawing if not creating contour
            if (
                not self.creating_contour
                and not self.scaling_active
                and not self.moving_active
                and not self.rotating_active
            ):
                self.drawing = True
                self.lasso_points = [self.mapToScene(event.pos())]

            # Remove any temporary items
            if self.rect_item:
                self.scene.removeItem(self.rect_item)
                self.rect_item = None
            if hasattr(self, "lasso_item") and self.lasso_item:
                self.scene.removeItem(self.lasso_item)
                self.lasso_item = None

            # --- Handle contour creation ---
            if self.creating_contour:
                scene_pos = self.mapToScene(event.pos())
                x, y = int(scene_pos.x()), int(scene_pos.y())
                x, y = self.clamp_to_image(x, y)
                self.freehand_drawing = True
                self.last_freehand_point = (x, y)

                # Start creating contour if not already
                if not self.currently_creating_contour:
                    self.currently_creating_contour = True
                    self.freehand_drawing = True
                    self.last_freehand_point = (x, y)

                # --- Check for polygon closure ---
                if self.hovering_start_point and len(self.new_contour_points) >= 3:
                    if self.contour_creation_undo_stack is not None:
                        if not self.contour_creation_undo_stack:  # <-- add this
                            self.contour_creation_undo_stack.append([])
                            self._clear_redo()
                        self.contour_creation_undo_stack[-1].append(
                            self.new_contour_points.copy()
                        )
                        self.contour_creation_redo_stack.clear()

                    if not self.contours:
                        self.contours.append(np.empty((0, 1, 2), dtype=np.int32))

                    # Now safely update
                    self.contours[-1] = np.array(
                        self.new_contour_points, dtype=np.int32
                    ).reshape((-1, 1, 2))

                    self.new_contour_points.clear()
                    self.currently_creating_contour = False
                    self.hovering_start_point = False

                    self.undo_stack.append([c.copy() for c in self.contours])
                    self.contours.append(np.empty((0, 1, 2), dtype=np.int32))
                    self.contour_creation_undo_stack.append([])
                    self._clear_redo()

                    self.update_display()
                    return

                # --- AOI check ---
                current_aoi = self.aois_dict.get(self.current_image_path, [])
                if current_aoi and len(current_aoi) >= 3:
                    aoi_polygon = QPolygonF([QPointF(px, py) for px, py in current_aoi])
                else:
                    aoi_polygon = None

                if aoi_polygon is None or aoi_polygon.containsPoint(
                    QPointF(x, y), QtCore.Qt.OddEvenFill
                ):
                    # --- Push undo state exactly once ---
                    if self.contour_creation_undo_stack is not None:
                        if not self.contour_creation_undo_stack:  # <-- add this
                            self.contour_creation_undo_stack.append([])
                        self.contour_creation_undo_stack[-1].append(
                            self.new_contour_points.copy()
                        )
                        self.contour_creation_redo_stack.clear()

                    # --- Add point for freehand or click ---
                    self.new_contour_points.append((x, y))
                    if not self.contours:
                        self.contours.append(np.empty((0, 1, 2), dtype=np.int32))

                    # Now safely update
                    self.contours[-1] = np.array(
                        self.new_contour_points, dtype=np.int32
                    ).reshape((-1, 1, 2))
                else:
                    print("Contour point outside AOI; ignoring")

                # Update display
                self.update_display()

        elif event.button() == QtCore.Qt.RightButton:
            self.pan_active = True
            self.last_mouse_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

        super().mousePressEvent(event)

    def move_selected_points(self):
        if (
            not self.selected_points
            or self.move_start_mouse_pos is None
            or self.mouse_pos is None
        ):
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
        self.hovering_start_point = False
        self.mouse_pos = self.mapToScene(event.pos())

        if self.creating_contour and self.new_contour_points:
            scene_pos = self.mapToScene(event.pos())
            mx, my = int(scene_pos.x()), int(scene_pos.y())
            mx, my = self.clamp_to_image(mx, my)

            sx, sy = self.new_contour_points[0]
            dist = np.hypot(mx - sx, my - sy)

            if dist <= self.start_point_radius and len(self.new_contour_points) >= 3:
                self.hovering_start_point = True

        if self.creating_contour and self.freehand_drawing:
            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            x, y = self.clamp_to_image(x, y)

            if self.last_freehand_point is not None:
                lx, ly = self.last_freehand_point
                dist = np.hypot(x - lx, y - ly)

                if dist >= self.freehand_min_dist:
                    self.last_freehand_point = (x, y)

                    if self.contour_creation_undo_stack is not None:
                        self.contour_creation_undo_stack[-1].append(
                            self.new_contour_points.copy()
                        )
                        self.contour_creation_redo_stack.clear()

                    self.new_contour_points.append((x, y))
                    if not self.contours:
                        self.contours.append(np.empty((0, 1, 2), dtype=np.int32))

                    # Now safely update
                    self.contours[-1] = np.array(
                        self.new_contour_points, dtype=np.int32
                    ).reshape((-1, 1, 2))

                    self.update_display()
            return

        if self.drawing:
            point = self.mouse_pos
            self.lasso_points.append(point)
            if hasattr(self, "lasso_item") and self.lasso_item:
                self.scene.removeItem(self.lasso_item)
            polygon = QtGui.QPolygonF(self.lasso_points)
            pen = QtGui.QPen(QtGui.QColor("red"))
            pen.setWidth(self.line_thickness)
            pen.setCosmetic(True)
            self.lasso_item = self.scene.addPolygon(polygon, pen)
        elif self.pan_active:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
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

        if (
            self.creating_contour
            and self.freehand_drawing
            and event.button() == QtCore.Qt.LeftButton
        ):
            self.freehand_drawing = False

            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            x, y = self.clamp_to_image(x, y)

            # --- Close polygon if hovering over start point ---
            if self.hovering_start_point and len(self.new_contour_points) >= 3:
                if self.contour_creation_undo_stack is not None:
                    if not self.contour_creation_undo_stack:  # <-- add this
                        self.contour_creation_undo_stack.append([])
                    self.contour_creation_undo_stack[-1].append(
                        self.new_contour_points.copy()
                    )
                    self.contour_creation_redo_stack.clear()

                if not self.contours:
                    self.contours.append(np.empty((0, 1, 2), dtype=np.int32))

                # Now safely update
                self.contours[-1] = np.array(
                    self.new_contour_points, dtype=np.int32
                ).reshape((-1, 1, 2))
                self.new_contour_points.clear()
                self.currently_creating_contour = False
                self.hovering_start_point = False

                self.contours.append(np.empty((0, 1, 2), dtype=np.int32))
                self.contour_creation_undo_stack.append([])

            self.last_freehand_point = None
            self.update_display()
            return

        if self.drawing and event.button() == QtCore.Qt.LeftButton:
            self.drawing = False

            # If fewer than 3 points, treat it as a click: deselect all
            if hasattr(self, "lasso_points") and len(self.lasso_points) < 3:
                self.selected_points.clear()

            elif hasattr(self, "lasso_points") and len(self.lasso_points) >= 3:
                polygon = QtGui.QPolygonF(self.lasso_points)
                self.select_points_in_polygon(polygon)

            if hasattr(self, "lasso_item") and self.lasso_item:
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
        self._clear_redo()
        self.contours = new_contours
        self.selected_points.clear()

    def remove_small_contours(self):
        """
        Automatically remove any contour with fewer than 3 points.
        Only runs when NOT creating a contour.
        """
        if self.creating_contour or self.currently_creating_contour:
            return

        self.contours = [cnt for cnt in self.contours if len(cnt) >= 3]

    def rotate_selected_points(self):
        if (
            not self.rotation_reference
            or not self.mouse_pos
            or self.rotation_start_angle is None
        ):
            return

        img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
        current_vec = np.array(
            [self.mouse_pos.x() - img_center.x(), self.mouse_pos.y() - img_center.y()]
        )
        current_angle = np.arctan2(current_vec[1], current_vec[0])

        angle_delta = current_angle - self.rotation_start_angle

        for (c_idx, pt_idx), orig_pt in self.rotation_reference.items():
            cnt = self.contours_original_for_rotation[c_idx]

            # Compute centroid of the contour
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            ox, oy = orig_pt
            dx, dy = ox - cx, oy - cy
            radius = np.sqrt(dx**2 + dy**2)
            original_angle = np.arctan2(dy, dx)
            new_angle = original_angle + angle_delta

            new_x = cx + radius * np.cos(new_angle)
            new_y = cy + radius * np.sin(new_angle)
            new_x, new_y = self.clamp_to_image(new_x, new_y)

            self.contours[c_idx][pt_idx][0] = [int(new_x), int(new_y)]

    def union_selected_contours(self):
        # Find which contours are involved
        selected_contour_indices = {c_idx for (c_idx, _) in self.selected_points}

        if len(selected_contour_indices) < 2:
            return  # need at least two contours

        # Save undo state
        self.undo_stack.append([c.copy() for c in self.contours])
        self._clear_redo()

        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw filled selected contours onto mask
        for c_idx in selected_contour_indices:
            cv2.drawContours(mask, self.contours, c_idx, 255, thickness=cv2.FILLED)

        # Extract outer contour of union
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        union_contour = max(contours, key=cv2.contourArea)

        # Build new contour list:
        # - remove old selected contours
        # - add union contour
        new_contours = []
        for i, cnt in enumerate(self.contours):
            if i not in selected_contour_indices:
                new_contours.append(cnt)

        new_contours.append(union_contour)

        self.contours = new_contours
        self.selected_points.clear()
        self.update_display()

    def scale_selected_points(self):
        if (
            not self.selected_points
            or not self.mouse_pos
            or not self.scaling_initial_distance
        ):
            return

        img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
        mouse_vec = np.array(
            [self.mouse_pos.x() - img_center.x(), self.mouse_pos.y() - img_center.y()]
        )
        current_distance = np.linalg.norm(mouse_vec)

        # Relative scaling factor from initial mouse distance
        new_scale = current_distance / self.scaling_initial_distance

        updated_contours = set()

        for (c_idx, pt_idx), orig_pt in self.scaling_reference.items():
            cnt_orig = self.contours_original_for_scaling[c_idx]
            cnt = self.contours[c_idx]

            # Compute contour centroid from original
            M = cv2.moments(cnt_orig)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

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
    def __init__(self, image_dict, loop, line_thickness=2, detected_contours=None):
        """
        image_dict: dict[str, np.ndarray]
            Keys are image identifiers (usually original paths),
            values are numpy images already loaded in memory.
        """
        super().__init__()
        self.loop = loop
        self.close_flag = False
        self.image_dict = image_dict
        self.image_keys = list(image_dict.keys())
        self.line_thickness = line_thickness
        self.index = 0
        self.results = {k: None for k in self.image_keys}  # stores contours per image
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

        self.finish_btn = QtWidgets.QPushButton("Finish")
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
        self.finish_btn.clicked.connect(self.finish_editing)

        finish_layout = QtWidgets.QHBoxLayout()
        finish_layout.addWidget(self.finish_btn, stretch=1)

        # Status label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)

        # Add button layout and status label to main layout
        self.layout.addWidget(self.status_label)
        self.layout.addLayout(self.button_layout)
        self.layout.addLayout(finish_layout)

        # Initialize ContourEditorView (empty for now)
        self.editor_view = None

        # Load first image
        self.load_image(self.index)

        self.showMaximized()

        def focus_window():
            self.raise_()  # bring window to top
            self.activateWindow()  # make it the active window
            self.setFocus(QtCore.Qt.ActiveWindowFocusReason)

        # Force reload after window has fully shown
        QtCore.QTimer.singleShot(0, focus_window)

        QtCore.QTimer.singleShot(0, lambda: self.change_image(0, force=True))

    def save_current_polygon(self):
        key = self.image_keys[self.index]
        if self.editor_view is not None:
            self.results[key] = [
                c.copy() for c in self.editor_view.get_edited_contours()
            ]

    def finish_editing(self):
        """Called when Finish button is clicked"""
        self.save_current_polygon()
        self.loop.quit()  # stop local loop
        self.close_flag = True
        self.close()

    def closeEvent(self, event):
        """Called when the window is closed"""
        if getattr(self, "close_flag", False):
            # Finish button triggered — just close window, do not exit program
            event.accept()
        else:
            # User clicked X — fully exit program
            print("Window closed — exiting program.")
            QtWidgets.QApplication.quit()
            sys.exit(0)

    def update_navigation_buttons(self):
        self.prev_btn.setEnabled(self.index > 0)
        self.next_btn.setEnabled(self.index < len(self.image_keys) - 1)

    def load_image(self, idx):
        key = self.image_keys[idx]
        detected = self.detected_contours.get(key, [])
        img = self.image_dict[key]

        if self.editor_view is not None:
            # Save contours of current image
            self.results[self.image_keys[self.index]] = (
                self.editor_view.get_edited_contours()
            )

            # Reset editor state for new image
            self.editor_view.image = img.copy()
            saved_contours = self.results.get(key)

            if saved_contours is not None:
                # User has visited this image before (even if they deleted everything)
                self.editor_view.contours = [c.copy() for c in saved_contours]
            else:
                # First visit → initialize from detector
                self.editor_view.contours = [c.copy() for c in detected]

            self.editor_view.original_contours = [
                c.copy() for c in self.editor_view.contours
            ]

            # ===== RESET TO SELECTION MODE HERE =====
            self.editor_view.selected_points.clear()
            self.editor_view.creating_contour = False
            self.editor_view.currently_creating_contour = False
            self.editor_view.scaling_active = False
            self.editor_view.moving_active = False
            self.editor_view.rotating_active = False
            self.editor_view.mouse_pos = None
            self.editor_view.scaling_reference = {}
            self.editor_view.rotation_reference = {}
            self.editor_view.contours_original_for_move = None
            self.editor_view.contours_original_for_rotation = None
            self.editor_view.rotation_start_angle = None
            self.editor_view.current_mode = "Selection"
            self.editor_view.resetTransform()
            self.editor_view.initial_fit_done = False
            self.editor_view.update_display()
        else:
            # First-time setup
            self.editor_view = ContourEditorView(
                img.copy(),
                contours=[],
                line_thickness=self.line_thickness,
                current_image_path=key,
                detected_contours=detected,
            )
            self.layout.insertWidget(0, self.editor_view, stretch=1)

        self.index = idx
        self.update_status()
        self.update_navigation_buttons()

        # Fit image to view
        QtCore.QTimer.singleShot(0, self.editor_view._initial_center_and_fit)
        self.editor_view.viewport().setFocus(QtCore.Qt.ActiveWindowFocusReason)

    def change_image(self, delta, force=False):
        new_index = max(0, min(len(self.image_keys) - 1, self.index + delta))
        if force or new_index != self.index:
            self.load_image(new_index)

    def update_status(self):
        self.status_label.setText(f"Image {self.index + 1} / {len(self.image_keys)}")

    def get_results(self):
        # Save the currently edited image
        if self.editor_view is not None:
            self.results[self.image_keys[self.index]] = (
                self.editor_view.get_edited_contours()
            )
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

    loop = QtCore.QEventLoop()

    editor_widget = MultiImageContourEditor(
        image_dict,
        loop,
        line_thickness=line_thickness,
        detected_contours=detected_contours,
    )
    editor_widget.setWindowTitle(
        "Multi-Image Contour Editor (C=Create Contour, D=Delete, ctrl+Z=Undo, ctrl+Y=Redo, S=Scale, M=Move, R=Rotate, U = Union of selected contours, esc = Return to Selection Mode)"
    )
    editor_widget.setWindowFlags(
        editor_widget.windowFlags() | QtCore.Qt.WindowStaysOnTopHint
    )
    editor_widget.showMaximized()
    editor_widget.setWindowFlags(
        editor_widget.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint
    )
    editor_widget.showMaximized()

    loop.exec_()

    return editor_widget.get_results()
