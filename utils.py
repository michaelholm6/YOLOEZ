import cv2
import ctypes
import tkinter as tk
from tkinter import messagebox
from PyQt5 import QtWidgets, QtGui

def resize_for_display(img, max_width, max_height):
    h, w = img.shape[:2]
    scale = 1.0
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img, scale

def get_screen_size(scale=0.9):
    user32 = ctypes.windll.user32
    screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return int(screen_width * scale), int(screen_height * scale)

def show_instructions(message, title="Instructions"):
    root = tk.Tk()
    root.withdraw()  # Hide main window
    messagebox.showinfo(title, message)
    root.destroy()
    
def make_label_with_tooltip(parent, text, tooltip_text):
    container = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    font = QtGui.QFont()
    font.setPointSize(16)

    label = QtWidgets.QLabel(text)
    label.setFont(font)
    label.setToolTip(tooltip_text)

    icon_label = QtWidgets.QLabel()
    icon_pix = parent.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
    icon_label.setPixmap(icon_pix.pixmap(16, 16))
    icon_label.setToolTip(tooltip_text)

    layout.addWidget(label)
    layout.addWidget(icon_label)
    return container