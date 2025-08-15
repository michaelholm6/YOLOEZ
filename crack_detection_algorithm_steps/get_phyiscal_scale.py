import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui

class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(QtCore.QPoint)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(event.pos())
        super().mousePressEvent(event)

class ScaleBarPicker(QtWidgets.QWidget):
    def __init__(self, image, line_thickness, supress_instructions=False):
        super().__init__()
        
        self.setWindowTitle("Draw Scale Bar")
        self.image_orig = image.copy()
        self.supress_instructions = supress_instructions
        self.line_thickness = line_thickness

        self.scale_bar_pts = []

        self.label = ClickableLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.label.clicked.connect(self.on_label_clicked)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        if not self.supress_instructions:
            QtWidgets.QMessageBox.information(self, "Instructions",
                "Click exactly two points on the image to define the scale bar.\n"
                "Press 'Esc' to cancel and close the application.\n")

        self.showMaximized()
        self.update_display()
        
    def keyPressEvent(self, event: QtGui.QKeyEvent):
            if event.key() == QtCore.Qt.Key_Escape:
                sys.exit(0)  # quit python script entirely
            else:
                super().keyPressEvent(event)

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def update_display(self):
        label_w, label_h = self.label.width(), self.label.height()
        h_orig, w_orig = self.image_orig.shape[:2]

        scale_w = label_w / w_orig if w_orig > 0 else 1
        scale_h = label_h / h_orig if h_orig > 0 else 1
        self.scale_factor = min(scale_w, scale_h)

        self.new_w = max(1, int(w_orig * self.scale_factor))
        self.new_h = max(1, int(h_orig * self.scale_factor))

        resized_img = cv2.resize(self.image_orig, (self.new_w, self.new_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        qimg = QtGui.QImage(img_rgb.data, self.new_w, self.new_h, 3 * self.new_w, QtGui.QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(qimg)

        # Draw points & line on pixmap copy
        pixmap_copy = self.pixmap.copy()
        painter = QtGui.QPainter(pixmap_copy)
        pen = QtGui.QPen(QtCore.Qt.red)
        pen.setWidth(4)
        painter.setPen(pen)

        for pt in self.scale_bar_pts:
            painter.drawEllipse(pt, self.line_thickness, self.line_thickness)
        if len(self.scale_bar_pts) == 2:
            painter.drawLine(self.scale_bar_pts[0], self.scale_bar_pts[1])
        painter.end()

        self.label.setPixmap(pixmap_copy)

    def on_label_clicked(self, pos):
        label_w, label_h = self.label.width(), self.label.height()
        offset_x = (label_w - self.new_w) // 2
        offset_y = (label_h - self.new_h) // 2

        # Check if click is inside the image
        if (offset_x <= pos.x() <= offset_x + self.new_w) and (offset_y <= pos.y() <= offset_y + self.new_h):
            # Adjust pos relative to the image
            img_x = pos.x() - offset_x
            img_y = pos.y() - offset_y

            self.scale_bar_pts.append(QtCore.QPoint(img_x, img_y))
            self.update_display()

            if len(self.scale_bar_pts) == 2:
                QtCore.QTimer.singleShot(100, self.ask_real_length)

    def ask_real_length(self):
        length, ok = QtWidgets.QInputDialog.getDouble(
            self, "Scale Bar Length",
            "Enter real-world length of the scale bar (in millimeters):",
            0.0,
            0.0001,
            1e6,
            4
        )

        if ok and length > 0:
            p1, p2 = self.scale_bar_pts
            x1, y1 = p1.x(), p1.y()
            x2, y2 = p2.x(), p2.y()

            px1, py1 = x1 / self.scale_factor, y1 / self.scale_factor
            px2, py2 = x2 / self.scale_factor, y2 / self.scale_factor

            pixel_length = np.hypot(px2 - px1, py2 - py1)
            um_per_pixel = (length * 1000) / pixel_length

            self.um_per_pixel = um_per_pixel
            self.close()
        else:
            self.scale_bar_pts = []
            self.update_display()

def get_scale_from_user(image, line_thickness, show_instructions=True):
    app = QtWidgets.QApplication.instance()
    close_app_after = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        close_app_after = True

    picker = ScaleBarPicker(image, line_thickness, show_instructions)
    picker.show()
    app.exec_()

    scale = getattr(picker, "um_per_pixel", None)

    if close_app_after:
        app.quit()

    if scale is None:
        raise ValueError("Scale bar was not defined.")

    return scale