# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import sys
import threading
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
from ultralytics import YOLO
import torch
from pyqtgraph.exporters import ImageExporter
import os
import numpy as np
from datetime import datetime
import math
from utils import show_error_window
from PyQt5 import QtCore


def run_training(dataset_yaml, model_save_dir, model_size, task="detection", prev_model_path=None):

    if task == "segmentation":
        default_model_path = f"yolo11{model_size}-seg.pt"
    else:
        default_model_path = f"yolo11{model_size}.pt"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_text = "GPU" if "cuda" in device else "CPU"
    print(f"⚡ Using device: {device}")

    # --- Handle previous model ---
    if prev_model_path:
        if not os.path.isfile(prev_model_path):
            show_error_window(f"The specified previous model path does not exist:\n{prev_model_path}")
            return
        model_path = prev_model_path
        print(f"✅ Continuing training from previous model: {model_path}")
    else:
        model_path = default_model_path
        print(f"ℹ Starting training from default weights: {model_path}")


    # ---------------- GUI ----------------
    class TrainingGUI(QtWidgets.QWidget):
        save_and_close_requested = QtCore.pyqtSignal()
        def __init__(self, device_text='Device: CPU'):
            super().__init__()
            self.setWindowTitle("YOLO Training Monitor")
            self.save_and_close_requested.connect(self._save_and_close)

            layout = QtWidgets.QVBoxLayout(self)

            def make_label_with_tooltip(text, tooltip):
                container = QtWidgets.QWidget()
                hlayout = QtWidgets.QHBoxLayout(container)
                hlayout.setContentsMargins(0, 0, 0, 0)
                hlayout.setSpacing(4)

                label = QtWidgets.QLabel(text)
                label.setObjectName("value_label")

                icon_label = QtWidgets.QLabel()
                icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
                icon_label.setPixmap(icon_pix.pixmap(14, 14))
                icon_label.setToolTip(tooltip)

                hlayout.addWidget(label)
                hlayout.addWidget(icon_label)
                hlayout.addStretch()
                return container, label

            self.device_label = QtWidgets.QLabel()
            self.device_label.setText(f"Training on {device_text.upper()}")
            font = QtGui.QFont()
            font.setPointSize(12)
            font.setBold(True)
            self.device_label.setFont(font)
            self.device_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(self.device_label)

            # ===== Metric Labels =====
            metrics_layout = QtWidgets.QVBoxLayout()
            self.precision_container, self.precision_label = make_label_with_tooltip(
                "Precision: 0.0",
                " How reliable the model is when it says an object is present. Higher is better. Range is from 0 to 1."
            )
            self.recall_container, self.recall_label = make_label_with_tooltip(
                "Recall: 0.0",
                "How well the model finds all objects present. Higher is better. Range is from 0 to 1."
            )
            self.map_container, self.map_label = make_label_with_tooltip(
                "mAP0.5–0.95: 0.0",
                "How good the model is at locating and classifying objects. Higher is better. Range is from 0 to 1."
            )
            for c in [self.precision_container, self.recall_container, self.map_container]:
                metrics_layout.addWidget(c)
            layout.addLayout(metrics_layout)

            # ===== Metrics Plot =====
            self.metrics_plot = pg.PlotWidget(title="Metrics")
            self.metrics_plot.setBackground('w')              # ← white background
            self.metrics_plot.getAxis('left').setPen('k')     # ← black axes
            self.metrics_plot.getAxis('bottom').setPen('k')
            self.metrics_plot.addLegend()
            self.metrics_plot.setLabel("left", "Value", color='k')
            self.metrics_plot.setLabel("bottom", "Epoch", color='k')
            layout.addWidget(self.metrics_plot)

            self.precision_curve = self.metrics_plot.plot(pen=pg.mkPen(color="r", width=2), name="Precision")
            self.recall_curve    = self.metrics_plot.plot(pen=pg.mkPen(color="b", width=2), name="Recall")
            self.map_curve       = self.metrics_plot.plot(pen=pg.mkPen(color="g", width=2), name="mAP0.5–0.95")

            # ===== Loss Labels =====
            loss_labels_layout = QtWidgets.QVBoxLayout()
            self.box_loss_container, self.box_loss_label = make_label_with_tooltip(
                "Validation Box Loss: 0.0",
                "This is a metric indicating how well the model predicts bounding boxes for detected objects. Lower is better. Range is from 0 to ∞."
            )
            self.cls_loss_container, self.cls_loss_label = make_label_with_tooltip(
                "Validation Classsification Loss: 0.0",
                "This is a metric indicating how well the model classifies detected objects. Lower is better. Range is from 0 to ∞."
            )
            self.dfl_loss_container, self.dfl_loss_label = make_label_with_tooltip(
                "Validation Distiribution Focal Loss: 0.0",
                "This is a metric indicating how well the predicted bounding box distributions align with the ground truth. Lower is better. Range is from 0 to ∞."
            )
            for c in [self.box_loss_container, self.cls_loss_container, self.dfl_loss_container]:
                loss_labels_layout.addWidget(c)
            layout.addLayout(loss_labels_layout)

            # ===== Loss Plot =====
            self.loss_plot = pg.PlotWidget(title="Validation Losses")
            self.loss_plot.setBackground('w')                  # ← white background
            self.loss_plot.getAxis('left').setPen('k')
            self.loss_plot.getAxis('bottom').setPen('k')
            self.loss_plot.addLegend()
            self.loss_plot.setLabel("left", "Loss (log scale)", color='k')
            self.loss_plot.setLabel("bottom", "Epoch", color='k')
            self.loss_plot.setLogMode(y=True)
            layout.addWidget(self.loss_plot)

            self.box_loss_curve = self.loss_plot.plot(pen=pg.mkPen(color="y", width=2), name="Box Loss")
            self.cls_loss_curve = self.loss_plot.plot(pen=pg.mkPen(color="c", width=2), name="Cls Loss")
            self.dfl_loss_curve = self.loss_plot.plot(pen=pg.mkPen(color="m", width=2), name="DFL Loss")

            # ===== Autoscale Buttons =====
            autoscale_layout = QtWidgets.QHBoxLayout()
            layout.addLayout(autoscale_layout)
            self.metrics_autoscale_btn = QtWidgets.QPushButton("Autoscale Metrics Plot")
            self.metrics_autoscale_btn.clicked.connect(lambda: self.autoscale_plot(self.metrics_plot, 'metrics'))
            autoscale_layout.addWidget(self.metrics_autoscale_btn)

            self.loss_autoscale_btn = QtWidgets.QPushButton("Autoscale Loss Plot")
            self.loss_autoscale_btn.clicked.connect(lambda: self.autoscale_plot(self.loss_plot, 'loss'))
            autoscale_layout.addWidget(self.loss_autoscale_btn)

            # ===== Stop Button =====
            self.stop_requested = False
            self.stop_button = QtWidgets.QPushButton("Stop Training")
            self.stop_button.clicked.connect(self.request_stop)
            layout.addWidget(self.stop_button)

            # ===== Buffers =====
            self.epochs = []
            self.precision_data, self.recall_data, self.map_data = [], [], []
            self.box_loss_data, self.cls_loss_data, self.dfl_loss_data = [], [], []

            # ===== Selected Lines Per Plot =====
            self.selected_metrics_line = None
            self.selected_loss_line = None

            # ===== Markers =====
            self.marker_metrics = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('k'))
            self.marker_metrics_text = pg.TextItem(anchor=(0,1), color='k')
            self.metrics_plot.addItem(self.marker_metrics)
            self.metrics_plot.addItem(self.marker_metrics_text)
            self.marker_metrics.setVisible(False)
            self.marker_metrics_text.setVisible(False)

            self.marker_loss = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('k'))
            self.marker_loss_text = pg.TextItem(anchor=(0,1), color='k')
            self.loss_plot.addItem(self.marker_loss)
            self.loss_plot.addItem(self.marker_loss_text)
            self.marker_loss.setVisible(False)
            self.marker_loss_text.setVisible(False)

            # ===== Connect Clicks / Moves =====
            self.metrics_plot.scene().sigMouseClicked.connect(lambda evt: self.on_click(evt, self.metrics_plot))
            self.metrics_plot.scene().sigMouseMoved.connect(lambda evt: self.on_mouse_move(evt, self.metrics_plot))
            self.loss_plot.scene().sigMouseClicked.connect(lambda evt: self.on_click(evt, self.loss_plot))
            self.loss_plot.scene().sigMouseMoved.connect(lambda evt: self.on_mouse_move(evt, self.loss_plot))

        # ---------------- Methods ----------------
        def request_stop(self):
            self.stop_requested = True
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stopping. Please wait a moment...")
            
        def _save_and_close(self):
            # Force a full redraw before export
            QtWidgets.QApplication.processEvents()

            #self.save_plots(model_save_dir)

            #QtWidgets.QApplication.processEvents()
            self.close()

        def update_metrics(self, epoch, precision, recall, map_value, box_loss, cls_loss, dfl_loss):
            self.epochs.append(epoch)
            self.precision_data.append(precision)
            self.recall_data.append(recall)
            self.map_data.append(map_value)
            self.box_loss_data.append(box_loss)
            self.cls_loss_data.append(cls_loss)
            self.dfl_loss_data.append(dfl_loss)

            # Update labels
            self.precision_label.setText(f"Precision: {precision:.4f}")
            self.recall_label.setText(f"Recall: {recall:.4f}")
            self.map_label.setText(f"mAP0.5–0.95: {map_value:.4f}")
            self.box_loss_label.setText(f"Validation Box Loss: {box_loss:.4f}")
            self.cls_loss_label.setText(f"Validation Classification Loss: {cls_loss:.4f}")
            self.dfl_loss_label.setText(f"Validation Distribution Focal Loss: {dfl_loss:.4f}")

            # Update plots
            self.precision_curve.setData(self.epochs, self.precision_data)
            self.recall_curve.setData(self.epochs, self.recall_data)
            self.map_curve.setData(self.epochs, self.map_data)
            self.box_loss_curve.setData(self.epochs, self.box_loss_data)
            self.cls_loss_curve.setData(self.epochs, self.cls_loss_data)
            self.dfl_loss_curve.setData(self.epochs, self.dfl_loss_data)
            
        def save_plots(self, model_save_dir):
            """Save metrics and loss plots to model_save_dir/plots with timestamp, autoscaled and clean."""
            plots_dir = os.path.join(model_save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%d%m%y__%H%M%S")

            # ----- Metrics Plot -----
            # Temporarily remove markers
            self.marker_metrics.setVisible(False)
            self.marker_metrics_text.setVisible(False)
            self.selected_metrics_line = None  # deselect any line
            self.metrics_plot.enableAutoRange()
            
            self.metrics_plot.repaint()
            QtWidgets.QApplication.processEvents()

            metrics_exporter = ImageExporter(self.metrics_plot.plotItem)
            metrics_file = os.path.join(plots_dir, f"metrics_plot_{timestamp}.png")
            metrics_exporter.export(metrics_file)
            print(f"✅ Saved metrics plot: {metrics_file}")

            # ----- Loss Plot -----
            # Temporarily remove markers
            self.marker_loss.setVisible(False)
            self.marker_loss_text.setVisible(False)
            self.selected_loss_line = None  # deselect any line 
            self.loss_plot.enableAutoRange()
            self.loss_plot.repaint()
            loss_exporter = ImageExporter(self.loss_plot.plotItem)
            
            loss_file = os.path.join(plots_dir, f"loss_plot_{timestamp}.png")
            loss_exporter.export(loss_file)
            print(f"✅ Saved loss plot: {loss_file}")

        def on_mouse_move(self, evt, plot):
            # Determine which set of lines and markers we're using
            if plot == self.metrics_plot:
                lines = [self.precision_curve, self.recall_curve, self.map_curve]
                marker, text_item = self.marker_metrics, self.marker_metrics_text
            else:
                lines = [self.box_loss_curve, self.cls_loss_curve, self.dfl_loss_curve]
                marker, text_item = self.marker_loss, self.marker_loss_text

            # Get mouse position in scene coordinates
            pos = evt
            if isinstance(evt, tuple):
                pos = evt[0]

            # Hide marker if mouse is outside plot
            if not plot.sceneBoundingRect().contains(pos):
                marker.setVisible(False)
                text_item.setVisible(False)
                return

            # Map mouse position to data coordinates
            mouse_point = plot.plotItem.vb.mapSceneToView(pos)
            x_mouse = mouse_point.x()
            y_mouse = mouse_point.y()

            closest_line = None
            closest_x = None
            closest_y = None
            min_dist = float("inf")

            # Collect all y-data for threshold calculation
            all_y_vals = []
            line_data = []
            for line in lines:
                x_data, y_data = line.getData()
                if x_data is not None and y_data is not None and len(x_data) > 0:
                    line_data.append((line, x_data, y_data))
                    all_y_vals.append(y_data)

            if not line_data:  # no data at all
                marker.setVisible(False)
                text_item.setVisible(False)
                return

            all_y_vals = np.concatenate(all_y_vals)
            y_range = max(all_y_vals) - min(all_y_vals) if len(all_y_vals) > 0 else 1.0
            selection_threshold = y_range * 0.1  # 10% of y-range

            # Check all lines for nearest actual data point
            for line, x_data, y_data in line_data:
                distances = np.hypot(x_data - x_mouse, y_data - y_mouse)
                idx = np.argmin(distances)
                dist = distances[idx]

                if dist < min_dist:
                    min_dist = dist
                    closest_line = line
                    closest_x = x_data[idx]
                    closest_y = y_data[idx]

            # Only show marker if near some point
            if closest_line is None or min_dist > selection_threshold:
                marker.setVisible(False)
                text_item.setVisible(False)
                return

            # Set marker
            marker.setData([closest_x], [closest_y])
            text_item.setText(f"x={closest_x:.2f}, y={closest_y:.4f}")
            self.update_marker_text_position(plot, text_item, closest_x, closest_y)

            marker.setVisible(True)
            text_item.setVisible(True)
            
        
        def autoscale_plot(self, plot, plot_type):
            """
            Autoscale the given plot while preserving the currently selected line and its marker.
            
            Args:
                plot (PlotWidget): The PyQtGraph plot to autoscale.
                plot_type (str): Either 'metrics' or 'loss', to determine which marker to update.
            """
            # Determine selected line and marker based on plot type
            if plot_type == 'metrics':
                selected_attr = 'selected_metrics_line'
                marker = self.marker_metrics
                marker_text = self.marker_metrics_text
            else:
                selected_attr = 'selected_loss_line'
                marker = self.marker_loss
                marker_text = self.marker_loss_text

            # Temporarily disable marker updates
            old_selected = getattr(self, selected_attr)
            setattr(self, selected_attr, None)

            # Autoscale the plot
            plot.enableAutoRange()

            # Restore selection
            setattr(self, selected_attr, old_selected)

            # Update marker if a line was selected
            if old_selected is not None:
                x_data, y_data = old_selected.getData()
                if len(x_data) > 0:
                    x_val, y_val = x_data[-1], y_data[-1]
                    marker.setData([x_val], [y_val])
                    marker_text.setText(f"x={x_val:.2f}, y={y_val:.4f}")
                    marker_text.setPos(x_val, y_val)
                    marker.setVisible(True)
                    marker_text.setVisible(True)

            
            
        def update_marker_text_position(
            self, plot, text_item, x_val, y_val, x_offset_frac=.05 , y_offset_frac=0.05
        ):
            """
            Place the TextItem near the point (x_val, y_val) in data coordinates.
            Defaults to right+above. Flips left or below if it would go outside the plot.
            Works for normal or log-scale plots.
            """
            vb = plot.getViewBox()
            (x_min, x_max), (y_min, y_max) = vb.viewRange()

            # Compute offsets in data coordinates
            x_offset = (x_max - x_min) * x_offset_frac
            y_offset = (y_max - y_min) * y_offset_frac

            # For log-scaled plot (loss), transform y_val
            y_val_plot = y_val

            # Default position: right + above
            x_text = x_val
            y_text = y_val_plot + y_offset

            # Margin: consider text width as fraction of plot width
            margin_frac = 0.1
            margin_x = (x_max - x_min) * margin_frac
            margin_y = (y_max - y_min) * margin_frac

            # Flip horizontally if too close to right edge
            if x_text + margin_x > x_max:
                x_text = x_val - x_offset  # move left
            # Flip vertically if too close to top edge
            if y_text + margin_y > y_max:
                y_text = y_val_plot - y_offset  # move below

            # Ensure it doesn't go below left or bottom edges
            if y_text - margin_y < y_min:
                y_text = y_val_plot + y_offset

            text_item.setPos(x_text, y_text)





    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    gui = TrainingGUI(device_text=device_text)
    gui.showMaximized()

    def train_thread():
        model = YOLO(model_path)

        def on_epoch_end(trainer):
            m = trainer.metrics

            gui.update_metrics(
                epoch=trainer.epoch,
                precision=m.get("metrics/precision(B)", 0.0),
                recall=m.get("metrics/recall(B)", 0.0),
                map_value=m.get("metrics/mAP50-95(B)", 0.0),
                box_loss=m.get("val/box_loss", 0.0),
                cls_loss=m.get("val/cls_loss", 0.0),
                dfl_loss=m.get("val/dfl_loss", 0.0),
            )

            if gui.stop_requested:
                trainer.stop = True


        model.add_callback("on_fit_epoch_end", on_epoch_end)

        model.train(
            data=dataset_yaml,
            epochs=int(1e10),
            device=device,
            project=model_save_dir,
            name="YOLO_EZ_training",
            patience=0,
            save=True
        )
        
        gui.save_and_close_requested.emit()

    thread = threading.Thread(target=train_thread)
    thread.start()
    app.exec_()
    thread.join()
