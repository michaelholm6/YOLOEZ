# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm

import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore


class ImageCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.setMinimumSize(400, 300)

    def set_image(self, path):
        pix = QtGui.QPixmap(path)
        if pix.isNull():
            self.pixmap = None
        else:
            self.pixmap = pix
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))

        if not self.pixmap:
            return

        scaled = self.pixmap.scaled(
            self.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)


class ImageViewer(QtWidgets.QWidget):
    def __init__(self, image_paths, loop=None):
        super().__init__()

        self.loop = loop
        self.close_flag = False

        self.image_paths = image_paths
        self.index = 0

        self.canvas = ImageCanvas(self)

        self.prev_btn = QtWidgets.QPushButton("◀ Previous")
        self.next_btn = QtWidgets.QPushButton("Next ▶")

        self.prev_btn.clicked.connect(lambda: self.change_image(-1))
        self.next_btn.clicked.connect(lambda: self.change_image(1))

        self.image_index_label = QtWidgets.QLabel("")
        self.image_index_label.setAlignment(QtCore.Qt.AlignCenter)
        font = self.image_index_label.font()
        font.setPointSize(14)
        self.image_index_label.setFont(font)

        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)

        self.finish_btn = QtWidgets.QPushButton("Finish")
        self.finish_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 16pt;
            }
            QPushButton:hover { background-color: #005a9e; }
            QPushButton:pressed { background-color: #004578; }
        """)
        self.finish_btn.clicked.connect(self.finish)

        finish_layout = QtWidgets.QHBoxLayout()
        finish_layout.addWidget(self.finish_btn, stretch=1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.image_index_label)
        layout.addLayout(nav_layout)
        layout.addLayout(finish_layout)

        self.setWindowTitle("Image Viewer")
        self.resize(1000, 700)

        self.load_current_image()

        self.showMaximized()  # show first
        self.raise_()
        self.activateWindow()

    def load_current_image(self):
        path = self.image_paths[self.index]
        self.canvas.set_image(path)
        self.update_label()
        self.update_buttons()

    def finish(self):
        """Finish button behavior"""
        self.close_flag = True
        if self.loop is not None:
            self.loop.quit()
        self.close()

    def closeEvent(self, event):
        if getattr(self, "close_flag", False):
            # Finish button triggered → just close this window
            event.accept()
        else:
            # User clicked X or pressed Esc → exit program
            QtWidgets.QApplication.quit()
            sys.exit(0)

    def update_label(self):
        self.image_index_label.setText(
            f"Image {self.index + 1} / {len(self.image_paths)}"
        )

    def update_buttons(self):
        self.prev_btn.setEnabled(self.index > 0)
        self.next_btn.setEnabled(self.index < len(self.image_paths) - 1)

    def change_image(self, delta):
        self.index = max(0, min(len(self.image_paths) - 1, self.index + delta))
        self.load_current_image()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            self.change_image(1)
        elif event.key() == QtCore.Qt.Key_Left:
            self.change_image(-1)
        elif event.key() == QtCore.Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


def view_images(image_paths):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    loop = QtCore.QEventLoop()

    win = ImageViewer(image_paths, loop=loop)
    win.show()

    # Block until Finish or user close
    loop.exec_()


if __name__ == "__main__":
    folder = "path/to/images"
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    images = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(exts)
    ]

    view_images(images)
