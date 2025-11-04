import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os
import tkinter as tk
from PyQt5.QtWidgets import QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QIcon

class BoundingBoxEditorView(QtWidgets.QGraphicsView):
    """
    Axis-aligned bounding box editor.

    Boxes are stored as numpy arrays of shape (4,1,2) with corner order:
      0: top-left
      1: top-right
      2: bottom-right
      3: bottom-left
    """
    def __init__(self, image, boxes=None, line_thickness=2, parent=None):
        super().__init__(parent)
        self.image = image
        self.boxes = [b.copy() for b in boxes] if boxes else []  # list of numpy arrays (4,1,2)
        self.selected_points = set()  # set of (box_idx, corner_idx)
        self.creating_box = False
        self.first_corner = None  # store first click
        self.currently_creating_box = False
        self.box_start_point = None
        self.rect_item = None

        self.current_mode = "Selection"
        self.line_thickness = line_thickness

        # Scene setup
        self.scene = QtWidgets.QGraphicsScene(self)
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(-10000, -10000, 20000, 20000)
        self.setScene(self.scene)

        # Rendering / interaction
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.zoom = 0.001
        self.pan_active = False
        self.last_mouse_pos = None
        self.undo_stack = []

        # initial render
        self.initial_fit_done = False
        self.update_display()
        QtCore.QTimer.singleShot(0, self._initial_center_and_fit)

    def _initial_center_and_fit(self):
        if not self.pixmap_item.pixmap() or self.initial_fit_done:
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)
        self.centerOn(self.pixmap_item)
        self.initial_fit_done = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self.initial_fit_done:
            self.resetTransform()
            self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)
            self.centerOn(self.pixmap_item)
            self.initial_fit_done = True

    def update_display(self):
        """Draw image and bounding boxes (with corner handles)."""
        img = self.image.copy()
        # Draw boxes
        for b_idx, box in enumerate(self.boxes):
            # box: np array shape (4,1,2) or (4,2)
            pts = [tuple(pt[0]) if pt.ndim == 2 else tuple(pt) for pt in box.reshape((-1,2))]
            # Ensure int coords
            pts_i = [(int(x), int(y)) for (x, y) in pts]
            # Draw rectangle (closed polyline)
            if len(pts_i) == 4:
                cv2.polylines(img, [np.array(pts_i)], isClosed=True, color=(0, 0, 255), thickness=self.line_thickness)
            # Draw corner handles
            for corner_idx, pt in enumerate(pts_i):
                if (b_idx, corner_idx) in self.selected_points:
                    cv2.circle(img, pt, self.line_thickness+2, (255, 0, 0), -1)  # selected -> blueish
                else:
                    cv2.circle(img, pt, max(2, int(self.line_thickness/2)), (0, 255, 0), -1)

        height, width, _ = img.shape
        qimage = QtGui.QImage(img.data, width, height, 3 * width, QtGui.QImage.Format_BGR888)
        self.pixmap_item.setPixmap(QtGui.QPixmap.fromImage(qimage))

        # Adjust scene rect
        scene_rect = self.pixmap_item.boundingRect()
        scene_rect = scene_rect.adjusted(-10000, -10000, 10000, 10000)
        self.scene.setSceneRect(scene_rect)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self.viewport())
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0)))
        font = QtGui.QFont("Arial", 20)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(50, 50, f"Mode: {self.current_mode}")
        painter.end()

    def keyPressEvent(self, event):
        # Undo
        if event.key() == QtCore.Qt.Key_U:
            if not self.creating_box and self.undo_stack:
                self.boxes = self.undo_stack.pop()
                self.selected_points.clear()
                self.update_display()
            return

        # Delete selected boxes (if any corner of a box is selected, delete whole box)
        if event.key() == QtCore.Qt.Key_D:
            if not self.creating_box:
                self.delete_selected_boxes()
                self.update_display()
            return

        # Quit (same behavior as before)
        if event.key() == QtCore.Qt.Key_Escape:
            sys.exit(0)

        # Toggle creation mode 'C'
        if event.key() == QtCore.Qt.Key_C:
            if self.creating_box:
                # Cancel creation mode if entered but not used
                self.creating_box = False
                self.currently_creating_box = False
                self.current_mode = "Selection"
                self.viewport().update()
                # Remove temp rect if exists
                if self.rect_item:
                    self.scene.removeItem(self.rect_item)
                    self.rect_item = None
            else:
                # Enter creation mode
                self.creating_box = True
                self.current_mode = "Box Creation"
                self.undo_stack.append([b.copy() for b in self.boxes])
                self.viewport().update()
            return

        # Toggle movement mode 'M'
        if event.key() == QtCore.Qt.Key_M:
            # Movement mode toggles on/off. When on, user can drag selected points / boxes.
            # We use the same M semantics as contour editor: start movement snapshot.
            if self.creating_box:
                return
            self.moving_active = not getattr(self, "moving_active", False)
            if self.moving_active:
                self.current_mode = "Moving"
                self.undo_stack.append([b.copy() for b in self.boxes])
                cursor_pos = QtGui.QCursor.pos()
                local_pos = self.mapFromGlobal(cursor_pos)
                self.move_start_mouse_pos = self.mapToScene(local_pos)
                self.boxes_original_for_move = [b.copy() for b in self.boxes]
            else:
                self.current_mode = "Selection"
                self.move_start_mouse_pos = None
                self.boxes_original_for_move = None
                self.selected_points.clear()
            self.update_display()
            return

        # Other keys pass to parent
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if getattr(self, "moving_active", False) and event.button() == QtCore.Qt.LeftButton:
            self.selected_points.clear()
            self.moving_active = False
            self.current_mode = "Selection"
            self.move_start_mouse_pos = None
            self.boxes_original_for_move = None
            self.update_display()
            return
        if event.button() == QtCore.Qt.LeftButton:
            # If we are in creation mode, start rectangle creation on left press.
            if self.creating_box:
                if self.first_corner is None:
                    # First click: set starting corner
                    self.first_corner = scene_pos
                    # Add temporary rectangle
                    if self.rect_item:
                        self.scene.removeItem(self.rect_item)
                    rect = QtCore.QRectF(self.first_corner, self.first_corner)
                    pen = QtGui.QPen(QtGui.QColor("red"))
                    pen.setWidth(self.line_thickness)
                    self.rect_item = self.scene.addRect(rect, pen)
                    self.current_mode = "Box Creation (first click)"
                else:
                    # Second click: finish box creation
                    rectf = self.rect_item.rect()
                    x1, y1 = int(rectf.left()), int(rectf.top())
                    x2, y2 = int(rectf.right()), int(rectf.bottom())
                    self.scene.removeItem(self.rect_item)
                    self.rect_item = None
                    # Create box corners: tl, tr, br, bl
                    tl = [x1, y1]
                    tr = [x2, y1]
                    br = [x2, y2]
                    bl = [x1, y2]
                    box = np.array([[[tl[0], tl[1]]],
                                    [[tr[0], tr[1]]],
                                    [[br[0], br[1]]],
                                    [[bl[0], bl[1]]]], dtype=np.int32)
                    # Only add if box is big enough
                    if abs(x2 - x1) >= 4 and abs(y2 - y1) >= 4:
                        self.boxes.append(box)
                    self.creating_box = False
                    self.first_corner = None
                    self.current_mode = "Selection"
                    self.update_display()
                return

            # If not creating, begin lasso-like drawing if not moving or panning
            if not getattr(self, "moving_active", False):
                # Begin lasso drawing (reuse drawing variable approach with polygon)
                self.drawing = True
                self.lasso_points = [scene_pos]
                # remove any temporary rect
                if self.rect_item:
                    self.scene.removeItem(self.rect_item)
                    self.rect_item = None
                return

            # If moving_active and left-click: start move (we already have move_start_mouse_pos set by key handler,
            # but also allow click to set it here)
            if getattr(self, "moving_active", False):
                # record current position as start (if not already)
                cursor_pos = QtGui.QCursor.pos()
                local_pos = self.mapFromGlobal(cursor_pos)
                self.move_start_mouse_pos = self.mapToScene(local_pos)
                self.boxes_original_for_move = [b.copy() for b in self.boxes]
                return

        elif event.button() == QtCore.Qt.RightButton:
            # Start panning
            self.pan_active = True
            self.last_mouse_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        # Updating temporary rect when creating box
        if self.creating_box and self.first_corner is not None:
            if self.rect_item:
                rect = QtCore.QRectF(self.first_corner, scene_pos).normalized()
                self.rect_item.setRect(rect)
            return

        # Lasso drawing
        if getattr(self, "drawing", False):
            self.lasso_points.append(scene_pos)
            if hasattr(self, 'lasso_item') and self.lasso_item:
                self.scene.removeItem(self.lasso_item)
            polygon = QtGui.QPolygonF(self.lasso_points)
            pen = QtGui.QPen(QtGui.QColor("red"))
            pen.setWidth(self.line_thickness)
            self.lasso_item = self.scene.addPolygon(polygon, pen)
            return

        # Panning
        if self.pan_active:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.viewport().update()
            return

        # Moving active (dragging selected corners / edges / boxes)
        if getattr(self, "moving_active", False) and getattr(self, "move_start_mouse_pos", None) is not None:
            # compute delta
            dx = scene_pos.x() - self.move_start_mouse_pos.x()
            dy = scene_pos.y() - self.move_start_mouse_pos.y()
            # apply movement logic based on selection
            self.boxes = [b.copy() for b in self.boxes_original_for_move]  # reset snapshot
            self._apply_move_delta(dx, dy)
            self.update_display()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if self.creating_box and self.currently_creating_box and event.button() == QtCore.Qt.LeftButton:
            # Finish creating box from start -> release
            self.currently_creating_box = False
            self.creating_box = False
            # remove temporary rect and convert to box if big enough
            if self.rect_item:
                rectf = self.rect_item.rect()
                # convert to ints and check size
                x1, y1 = int(rectf.left()), int(rectf.top())
                x2, y2 = int(rectf.right()), int(rectf.bottom())
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                self.scene.removeItem(self.rect_item)
                self.rect_item = None
                # Small boxes ignored
                if w >= 4 and h >= 4:
                    # Create box corners: tl, tr, br, bl
                    tl = [x1, y1]
                    tr = [x2, y1]
                    br = [x2, y2]
                    bl = [x1, y2]
                    box = np.array([[[tl[0], tl[1]]],
                                    [[tr[0], tr[1]]],
                                    [[br[0], br[1]]],
                                    [[bl[0], bl[1]]]], dtype=np.int32)
                    self.boxes.append(box)
                # done creation
            self.current_mode = "Selection"
            self.viewport().update()
            self.update_display()
            return

        # Finish lasso selection on left release
        if hasattr(self, 'drawing') and self.drawing and event.button() == QtCore.Qt.LeftButton:
            self.drawing = False
            if hasattr(self, 'lasso_item') and self.lasso_item:
                polygon = QtGui.QPolygonF(self.lasso_points)
                self.select_points_in_polygon(polygon)
                self.scene.removeItem(self.lasso_item)
                self.lasso_item = None
            self.update_display()
            return

        if event.button() == QtCore.Qt.RightButton:
            self.pan_active = False
            self.setCursor(QtCore.Qt.ArrowCursor)

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_factor = 1.2 if delta > 0 else 1 / 1.2
        self.zoom *= zoom_factor
        self.scale(zoom_factor, zoom_factor)

    def select_points_in_polygon(self, polygon):
        """Select corner points that fall inside the polygon."""
        self.selected_points.clear()
        for b_idx, box in enumerate(self.boxes):
            pts = box.reshape((-1,2))
            for corner_idx, (x, y) in enumerate(pts):
                pointf = QtCore.QPointF(x, y)
                if polygon.containsPoint(pointf, QtCore.Qt.OddEvenFill):
                    self.selected_points.add((b_idx, corner_idx))

    def get_edited_boxes(self):
        """Return boxes as a list of numpy arrays (same internal format)."""
        return self.boxes

    def delete_selected_boxes(self):
        """If any corner of a box is selected, delete the whole box."""
        if not self.selected_points:
            return
        boxes_to_keep = []
        for idx, box in enumerate(self.boxes):
            # if any selected point belongs to this box, skip it
            has_selected = any(bi == idx for (bi, ci) in self.selected_points)
            if not has_selected:
                boxes_to_keep.append(box)
        # save for undo
        self.undo_stack.append(self.boxes.copy())
        self.boxes = boxes_to_keep
        self.selected_points.clear()

    def _apply_move_delta(self, dx, dy):
        """
        Apply movement delta dx, dy to boxes according to selection rules:
         - one corner selected: move that corner to (x+dx, y+dy) and update adjacent corners
         - two adjacent corners selected (edge): move that edge in axis-aligned direction
         - otherwise: translate entire box
        """
        if not self.selected_points:
            return

        # Helper to clamp coordinates optionally to image bounds (not required, but could be added)
        def clamp_point(x, y):
            # keep inside image boundaries
            h, w = self.image.shape[:2]
            x_c = int(min(max(0, round(x)), w-1))
            y_c = int(min(max(0, round(y)), h-1))
            return x_c, y_c

        # Build mapping: box_idx -> list of selected corners
        sel_map = {}
        for b_idx, c_idx in self.selected_points:
            sel_map.setdefault(b_idx, []).append(c_idx)

        for b_idx, sel_corners in sel_map.items():
            box = self.boxes[b_idx]  # reference after we reset to original snapshot
            pts = box.reshape((-1,2)).astype(float)  # use float for arithmetic
            orig_pts = self.boxes_original_for_move[b_idx].reshape((-1,2)).astype(float)

            if len(sel_corners) == 1:
                # Move one corner: that corner moves to new pos; adjacent corners adjust to keep axis-aligned
                corner = sel_corners[0]
                # indices of corners
                # 0 tl, 1 tr, 2 br, 3 bl
                # adjacent corners indices:
                adj1 = (corner - 1) % 4
                adj2 = (corner + 1) % 4
                opp = (corner + 2) % 4

                # Move selected corner by dx/dy
                new_x = orig_pts[corner,0] + dx
                new_y = orig_pts[corner,1] + dy

                # Opposite corner stays fixed (orig_pts[opp])
                ox, oy = orig_pts[opp]

                # Determine new rectangle bounds by combining moved corner and opposite corner
                x_min = min(new_x, ox)
                x_max = max(new_x, ox)
                y_min = min(new_y, oy)
                y_max = max(new_y, oy)

                # Update corners in standard order (tl,tr,br,bl)
                new_pts = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ])

                # Write back (clamped to ints)
                for i in range(4):
                    xi, yi = clamp_point(new_pts[i,0], new_pts[i,1])
                    self.boxes[b_idx][i][0] = [xi, yi]

            elif len(sel_corners) == 2:
                c0, c1 = sel_corners[0], sel_corners[1]
                # Check adjacency (difference modulo 4 equals 1 or 3)
                if (c0 + 1) % 4 == c1 or (c1 + 1) % 4 == c0:
                    # adjacent corners => moving an edge
                    # find the shared edge orientation:
                    # edges: (0-1)=top, (1-2)=right, (2-3)=bottom, (3-0)=left
                    edge_indices = sorted([c0, c1])
                    # For top or bottom edge (indices 0&1 or 2&3) move vertically only
                    if set(edge_indices) == {0,1} or set(edge_indices) == {2,3}:
                        # vertical move only: compute dy from original average of those corners
                        new_y0 = orig_pts[c0,1] + dy
                        new_y1 = orig_pts[c1,1] + dy
                        # update both corners' y, and also update the opposite edge's corners to keep width same
                        # Determine x coords remain same as original
                        x0 = orig_pts[c0,0]; x1 = orig_pts[c1,0]
                        # Build new corners using min/max logic
                        y_min = min(new_y0, orig_pts[(edge_indices[0]+2) % 4,1])
                        y_max = max(new_y0, orig_pts[(edge_indices[0]+2) % 4,1])
                        new_pts = np.array([
                            [min(x0,x1), y_min],
                            [max(x0,x1), y_min],
                            [max(x0,x1), y_max],
                            [min(x0,x1), y_max],
                        ])
                        for i in range(4):
                            xi, yi = clamp_point(new_pts[i,0], new_pts[i,1])
                            self.boxes[b_idx][i][0] = [xi, yi]
                    else:
                        # left/right edge: move horizontally only
                        new_x0 = orig_pts[c0,0] + dx
                        new_x1 = orig_pts[c1,0] + dx
                        y0 = orig_pts[c0,1]; y1 = orig_pts[c1,1]
                        x_min = min(new_x0, orig_pts[(edge_indices[0]+2) % 4,0])
                        x_max = max(new_x0, orig_pts[(edge_indices[0]+2) % 4,0])
                        new_pts = np.array([
                            [x_min, min(y0,y1)],
                            [x_max, min(y0,y1)],
                            [x_max, max(y0,y1)],
                            [x_min, max(y0,y1)],
                        ])
                        for i in range(4):
                            xi, yi = clamp_point(new_pts[i,0], new_pts[i,1])
                            self.boxes[b_idx][i][0] = [xi, yi]
                else:
                    # Non-adjacent pair -> treat as whole box move
                    for i in range(4):
                        nx = orig_pts[i,0] + dx
                        ny = orig_pts[i,1] + dy
                        xi, yi = clamp_point(nx, ny)
                        self.boxes[b_idx][i][0] = [xi, yi]
            else:
                # Any other selection (3 or 4 corners or corners across different boxes) -> translate entire box
                for i in range(4):
                    nx = orig_pts[i,0] + dx
                    ny = orig_pts[i,1] + dy
                    xi, yi = clamp_point(nx, ny)
                    self.boxes[b_idx][i][0] = [xi, yi]

    # backward-compat helpers (MultiImageXXX depends on these names earlier)
    def get_edited_contours(self):
        # return same method name as earlier but now returns boxes
        return self.get_edited_boxes()

class MultiImageBoxEditor(QtWidgets.QWidget):
    def __init__(self, image_files, line_thickness=2):
        super().__init__()
        self.image_files = image_files
        self.line_thickness = line_thickness
        self.index = 0
        self.results = {path: [] for path in image_files}  # stores boxes per image (list of numpy arrays)

        # Layouts
        self.layout = QtWidgets.QVBoxLayout(self)
        self.button_layout = QtWidgets.QHBoxLayout()

        # Navigation buttons
        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.next_btn = QtWidgets.QPushButton("Next")
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

        # Editor view placeholder
        self.editor_view = None

        self.setWindowTitle("Multi-Image Bounding Box Editor")
        self.showMaximized()
        def focus_window():
            self.raise_()                 # bring window to top
            self.activateWindow()         # make it the active window
            self.setFocus(QtCore.Qt.ActiveWindowFocusReason)

        QtCore.QTimer.singleShot(0, focus_window)
        
        QtCore.QTimer.singleShot(0, lambda: self.change_image(0, force=True))

    def load_image(self, idx):
        path = self.image_files[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")

        # Save current boxes
        if self.editor_view is not None:
            self.results[self.image_files[self.index]] = self.editor_view.get_edited_boxes()
            self.layout.removeWidget(self.editor_view)
            self.editor_view.deleteLater()

        saved_boxes = self.results.get(path, [])
        # ensure saved_boxes are deep copies
        boxes_copy = [b.copy() for b in saved_boxes] if saved_boxes else []
        self.editor_view = BoundingBoxEditorView(
            img,
            boxes=boxes_copy,
            line_thickness=self.line_thickness
        )

        self.layout.insertWidget(0, self.editor_view, stretch=1)
        self.index = idx
        self.update_status()
        def focus_editor():
            self.editor_view.viewport().setFocus(QtCore.Qt.ActiveWindowFocusReason)
        QtCore.QTimer.singleShot(0, focus_editor)

        # fix centering after layout
        def center_when_ready():
            if self.editor_view.viewport().width() > 0 and self.editor_view.viewport().height() > 0:
                self.editor_view._initial_center_and_fit()
            else:
                QtCore.QTimer.singleShot(10, center_when_ready)
        QtCore.QTimer.singleShot(0, center_when_ready)

    def change_image(self, delta, force=False):
        new_index = max(0, min(len(self.image_files) - 1, self.index + delta))
        if force or new_index != self.index:
            self.load_image(new_index)

    def update_status(self):
        self.status_label.setText(f"Image {self.index + 1} / {len(self.image_files)}")

    def get_results(self):
        # Save current edited boxes
        if self.editor_view is not None:
            self.results[self.image_files[self.index]] = self.editor_view.get_edited_boxes()
        return self.results

def run_box_editor(image_files, line_thickness=2):
    """
    Launch bounding box editor for one or more images.

    Returns:
       dict: {image_path: [np.array(box), ...]} where each box is shape (4,1,2) and dtype int32
    """
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        created_app = True

    editor_widget = MultiImageBoxEditor(image_files, line_thickness=line_thickness)
    editor_widget.setWindowTitle("Bounding Box Editor (PyQt)")
    editor_widget.setWindowFlags(editor_widget.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
    editor_widget.showMaximized()

    if created_app:
        app.exec_()
    else:
        app.exec_()

    return editor_widget.get_results()
