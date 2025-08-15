import sys
import tkinter as tk
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QKeyEvent
from PyQt5.QtCore import Qt, QPoint, QTimer

def get_polygon_from_user(image, supress_instructions=False):
    # === Tkinter instructions window ===
    if not supress_instructions:
        root = tk.Tk()
        root.title("Instructions")

        # Your instructions text
        instructions = (
        "Instructions for Marking the Area of Interest:\n\n"
        "- You will now select your area of interest by drawing a polygon on the image.\n"
        "- The polygon should outline the area where you want to detect cracks.\n"
        "- Left click to add points and outline your polygon.\n"
        "- Press 'C' to close the polygon and exit the window.\n"
        "- Press 'R' to reset points.\n"
        "- Press 'Esc' to cancel and close the application.\n"
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
           

    # === PyQt Polygon Drawer ===
    class PolygonDrawer(QWidget):
        def __init__(self, image):
            super().__init__()
            self.orig_image = image
            self.orig_height, self.orig_width = image.shape[:2]
            self.polygon_points = []
            self.polygon_closed = False
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
            import cv2
            from PyQt5.QtGui import QImage

            if len(img.shape) == 2:
                qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(255, 255, 255))
            painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)

            if len(self.polygon_points) > 0:
                pen = QPen(QColor(0, 255, 0), 2)
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
            elif key == Qt.Key_Escape:
                self.polygon_points = []
                self.polygon_closed = False
                self.close()
                exit()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

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
        raise ValueError("At least 3 points are required to form a region of interest.")