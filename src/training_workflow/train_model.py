# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import sys
import traceback
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
from ultralytics import YOLO
import torch
import os
import numpy as np
from utils import ensure_app, show_error_window
from PyQt5 import QtCore


def run_training(
    dataset_yaml, model_save_dir, model_size, task="detection", prev_model_path=None
):
    """Launch the training GUI, spin up a background thread, and block until training completes.

    Args:
        dataset_yaml: Path to the YOLO dataset.yaml file.
        model_save_dir: Directory where the trained model will be saved.
        model_size: Single character size code (n/s/m/l/x) for the YOLO11 variant.
        task: "detection" or "segmentation".
        prev_model_path: Optional path to a .pt file to continue training from.
    """
    if task == "segmentation":
        default_model_path = f"yolo11{model_size}-seg.pt"
    else:
        default_model_path = f"yolo11{model_size}.pt"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_text = "GPU" if "cuda" in device else "CPU"
    print(f"Using device: {device}")

    if prev_model_path:
        if not os.path.isfile(prev_model_path):
            show_error_window(
                f"The specified previous model path does not exist:\n{prev_model_path}"
            )
            return None
        model_path = prev_model_path
        print(f"Continuing training from previous model: {model_path}")
    else:
        model_path = default_model_path
        print(f"Starting training from default weights: {model_path}")

    class TrainingGUI(QtWidgets.QWidget):
        save_and_close_requested = QtCore.pyqtSignal()
        metrics_updated = QtCore.pyqtSignal(
            int, float, float, float, float, float, float
        )

        def __init__(self, device_text="Device: CPU"):
            """Build the metrics/loss plots and all control widgets."""
            super().__init__()
            self.setWindowTitle("YOLO Training Monitor")
            self.save_and_close_requested.connect(self._save_and_close)
            self.metrics_updated.connect(self.update_metrics)

            layout = QtWidgets.QVBoxLayout(self)

            def make_label_with_tooltip(text, tooltip):
                container = QtWidgets.QWidget()
                hlayout = QtWidgets.QHBoxLayout(container)
                hlayout.setContentsMargins(0, 0, 0, 0)
                hlayout.setSpacing(4)

                label = QtWidgets.QLabel(text)
                label.setObjectName("value_label")

                icon_label = QtWidgets.QLabel()
                icon_pix = self.style().standardIcon(
                    QtWidgets.QStyle.SP_MessageBoxQuestion
                )
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

            metrics_layout = QtWidgets.QVBoxLayout()
            self.precision_container, self.precision_label = make_label_with_tooltip(
                "Precision: 0.0",
                " How reliable the model is when it says an object is present. Higher is better. Range is from 0 to 1.",
            )
            self.recall_container, self.recall_label = make_label_with_tooltip(
                "Recall: 0.0",
                "How well the model finds all objects present. Higher is better. Range is from 0 to 1.",
            )
            self.map_container, self.map_label = make_label_with_tooltip(
                "mAP0.5–0.95: 0.0",
                "How good the model is at locating and classifying objects. Higher is better. Range is from 0 to 1.",
            )
            for c in [
                self.precision_container,
                self.recall_container,
                self.map_container,
            ]:
                metrics_layout.addWidget(c)
            layout.addLayout(metrics_layout)

            self.metrics_plot = pg.PlotWidget(title="Metrics")
            self.metrics_plot.showGrid(x=True, y=True, alpha=0.3)
            self.metrics_plot.setBackground("w")
            self.metrics_plot.getAxis("left").setPen("k")
            self.metrics_plot.getAxis("bottom").setPen("k")
            self.metrics_plot.addLegend()
            self.metrics_plot.setLabel("left", "Value", color="k")
            self.metrics_plot.setLabel("bottom", "Epoch", color="k")
            layout.addWidget(self.metrics_plot)

            self.precision_curve = self.metrics_plot.plot(
                pen=pg.mkPen(color="r", width=2), name="Precision"
            )
            self.recall_curve = self.metrics_plot.plot(
                pen=pg.mkPen(color="b", width=2), name="Recall"
            )
            self.map_curve = self.metrics_plot.plot(
                pen=pg.mkPen(color="g", width=2), name="mAP0.5–0.95"
            )

            loss_labels_layout = QtWidgets.QVBoxLayout()
            self.box_loss_container, self.box_loss_label = make_label_with_tooltip(
                "Validation Box Loss: 0.0",
                "This is a metric indicating how well the model predicts bounding boxes for detected objects. Lower is better. Range is from 0 to ∞.",
            )
            self.cls_loss_container, self.cls_loss_label = make_label_with_tooltip(
                "Validation Classsification Loss: 0.0",
                "This is a metric indicating how well the model classifies detected objects. Lower is better. Range is from 0 to ∞.",
            )
            self.dfl_loss_container, self.dfl_loss_label = make_label_with_tooltip(
                "Validation Distiribution Focal Loss: 0.0",
                "This is a metric indicating how well the predicted bounding box distributions align with the ground truth. Lower is better. Range is from 0 to ∞.",
            )
            for c in [
                self.box_loss_container,
                self.cls_loss_container,
                self.dfl_loss_container,
            ]:
                loss_labels_layout.addWidget(c)
            layout.addLayout(loss_labels_layout)

            self.loss_plot = pg.PlotWidget(title="Validation Losses")
            self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
            self.loss_plot.setBackground("w")
            self.loss_plot.getAxis("left").setPen("k")
            self.loss_plot.getAxis("bottom").setPen("k")
            self.loss_plot.addLegend()
            self.loss_plot.setLabel("left", "Loss (log scale)", color="k")
            self.loss_plot.setLabel("bottom", "Epoch", color="k")
            self.loss_plot.setLogMode(y=True)
            layout.addWidget(self.loss_plot)

            self.box_loss_curve = self.loss_plot.plot(
                pen=pg.mkPen(color="k", width=2), name="Box Loss"
            )
            self.cls_loss_curve = self.loss_plot.plot(
                pen=pg.mkPen(color="c", width=2), name="Cls Loss"
            )
            self.dfl_loss_curve = self.loss_plot.plot(
                pen=pg.mkPen(color="m", width=2), name="DFL Loss"
            )

            autoscale_layout = QtWidgets.QHBoxLayout()
            layout.addLayout(autoscale_layout)
            self.metrics_autoscale_btn = QtWidgets.QPushButton("Autoscale Metrics Plot")
            self.metrics_autoscale_btn.clicked.connect(
                lambda: self.autoscale_plot(self.metrics_plot, "metrics")
            )
            autoscale_layout.addWidget(self.metrics_autoscale_btn)

            self.loss_autoscale_btn = QtWidgets.QPushButton("Autoscale Loss Plot")
            self.loss_autoscale_btn.clicked.connect(
                lambda: self.autoscale_plot(self.loss_plot, "loss")
            )
            autoscale_layout.addWidget(self.loss_autoscale_btn)

            self.stop_requested = False
            self.closing_programmatically = False
            self.stop_button = QtWidgets.QPushButton("Stop Training")
            self.stop_button.clicked.connect(self.request_stop)
            layout.addWidget(self.stop_button)

            self.epochs = []
            self.precision_data, self.recall_data, self.map_data = [], [], []
            self.box_loss_data, self.cls_loss_data, self.dfl_loss_data = [], [], []

            self.selected_metrics_line = None
            self.selected_loss_line = None

            self.marker_metrics = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("k"))
            self.marker_metrics_text = pg.TextItem(anchor=(0, 1), color="k")
            self.metrics_plot.addItem(self.marker_metrics)
            self.metrics_plot.addItem(self.marker_metrics_text)
            self.marker_metrics.setVisible(False)
            self.marker_metrics_text.setVisible(False)

            self.marker_loss = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("k"))
            self.marker_loss_text = pg.TextItem(anchor=(0, 1), color="k")
            self.loss_plot.addItem(self.marker_loss)
            self.loss_plot.addItem(self.marker_loss_text)
            self.marker_loss.setVisible(False)
            self.marker_loss_text.setVisible(False)

            self.metrics_plot.scene().sigMouseClicked.connect(
                lambda evt: self.on_click(evt, self.metrics_plot)
            )
            self.metrics_plot.scene().sigMouseMoved.connect(
                lambda evt: self.on_mouse_move(evt, self.metrics_plot)
            )
            self.loss_plot.scene().sigMouseClicked.connect(
                lambda evt: self.on_click(evt, self.loss_plot)
            )
            self.loss_plot.scene().sigMouseMoved.connect(
                lambda evt: self.on_mouse_move(evt, self.loss_plot)
            )

        def request_stop(self):
            """Signal the training thread to stop after the current epoch finishes."""
            self.stop_requested = True
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stopping. Please wait a moment...")

        def _save_and_close(self):
            """Flush pending GUI events then close the window.  Called via signal from the training thread."""
            self.closing_programmatically = True
            QtWidgets.QApplication.processEvents()
            self.close()

        def closeEvent(self, event):
            """
            Hard kill if the user clicks the window 'X' mid-training.
            No graceful shutdown — immediately terminates the process.

            A close requested by the program itself (training finished, or it
            failed) is a normal close: killing the process there would skip the
            caller's cleanup and completion message.
            """
            if self.closing_programmatically or self.stop_requested:
                event.accept()
                return

            os._exit(0)  # immediate, no cleanup, kills all threads

        def update_metrics(
            self, epoch, precision, recall, map_value, box_loss, cls_loss, dfl_loss
        ):
            """Append a new epoch's metrics and redraw all plot curves and value labels."""
            self.epochs.append(epoch)
            self.precision_data.append(precision)
            self.recall_data.append(recall)
            self.map_data.append(map_value)
            self.box_loss_data.append(box_loss)
            self.cls_loss_data.append(cls_loss)
            self.dfl_loss_data.append(dfl_loss)

            self.precision_label.setText(f"Precision: {precision:.4f}")
            self.recall_label.setText(f"Recall: {recall:.4f}")
            self.map_label.setText(f"mAP0.5–0.95: {map_value:.4f}")
            self.box_loss_label.setText(f"Validation Box Loss: {box_loss:.4f}")
            self.cls_loss_label.setText(
                f"Validation Classification Loss: {cls_loss:.4f}"
            )
            self.dfl_loss_label.setText(
                f"Validation Distribution Focal Loss: {dfl_loss:.4f}"
            )

            self.precision_curve.setData(self.epochs, self.precision_data)
            self.recall_curve.setData(self.epochs, self.recall_data)
            self.map_curve.setData(self.epochs, self.map_data)
            self.box_loss_curve.setData(self.epochs, self.box_loss_data)
            self.cls_loss_curve.setData(self.epochs, self.cls_loss_data)
            self.dfl_loss_curve.setData(self.epochs, self.dfl_loss_data)

        def on_mouse_move(self, evt, plot):
            """Snap a crosshair marker to the nearest data point within 10 screen pixels of the cursor."""
            if plot == self.metrics_plot:
                lines = [self.precision_curve, self.recall_curve, self.map_curve]
                marker, text_item = self.marker_metrics, self.marker_metrics_text
            else:
                lines = [self.box_loss_curve, self.cls_loss_curve, self.dfl_loss_curve]
                marker, text_item = self.marker_loss, self.marker_loss_text

            pos = evt
            if isinstance(evt, tuple):
                pos = evt[0]

            if not plot.sceneBoundingRect().contains(pos):
                marker.setVisible(False)
                text_item.setVisible(False)
                return

            vb = plot.plotItem.vb
            mouse_point = vb.mapSceneToView(pos)
            x_mouse = mouse_point.x()
            y_mouse = mouse_point.y()

            # Key: get pixel scaling
            dx_per_pixel, dy_per_pixel = vb.viewPixelSize()

            PIXEL_THRESHOLD = 10
            threshold_sq = PIXEL_THRESHOLD * PIXEL_THRESHOLD

            closest_line = None
            closest_x = None
            closest_y = None
            min_dist_sq = float("inf")

            found_any = False

            for line in lines:
                x_data, y_data = line.getData()
                if x_data is None or y_data is None or len(x_data) == 0:
                    continue

                found_any = True

                # --- VECTOR distance in "pixel space" ---
                dx = (x_data - x_mouse) / dx_per_pixel
                dy = (y_data - y_mouse) / dy_per_pixel

                dist_sq = dx * dx + dy * dy

                idx = np.argmin(dist_sq)
                if dist_sq[idx] < min_dist_sq:
                    min_dist_sq = dist_sq[idx]
                    closest_line = line
                    closest_x = x_data[idx]
                    closest_y = y_data[idx]

            if not found_any or closest_line is None or min_dist_sq > threshold_sq:
                marker.setVisible(False)
                text_item.setVisible(False)
                return

            marker.setData([closest_x], [closest_y])
            text_item.setText(self._format_point(plot, closest_x, closest_y))
            self.update_marker_text_position(plot, text_item, closest_x, closest_y)

            marker.setVisible(True)
            text_item.setVisible(True)

        def on_click(self, evt, plot):
            """Pin the marker to the curve nearest the click, or clear it on empty space."""
            if plot == self.metrics_plot:
                lines = [self.precision_curve, self.recall_curve, self.map_curve]
                marker, text_item = self.marker_metrics, self.marker_metrics_text
                selected_attr = "selected_metrics_line"
            else:
                lines = [self.box_loss_curve, self.cls_loss_curve, self.dfl_loss_curve]
                marker, text_item = self.marker_loss, self.marker_loss_text
                selected_attr = "selected_loss_line"

            pos = getattr(evt, "scenePos", None)
            pos = pos() if callable(pos) else evt
            if isinstance(pos, tuple):
                pos = pos[0]

            if not plot.sceneBoundingRect().contains(pos):
                return

            vb = plot.plotItem.vb
            mouse_point = vb.mapSceneToView(pos)
            dx_per_pixel, dy_per_pixel = vb.viewPixelSize()

            PIXEL_THRESHOLD = 10
            closest = None
            min_dist_sq = float("inf")

            for line in lines:
                x_data, y_data = line.getData()
                if x_data is None or y_data is None or len(x_data) == 0:
                    continue

                dx = (x_data - mouse_point.x()) / dx_per_pixel
                dy = (y_data - mouse_point.y()) / dy_per_pixel
                dist_sq = dx * dx + dy * dy

                idx = np.argmin(dist_sq)
                if dist_sq[idx] < min_dist_sq:
                    min_dist_sq = dist_sq[idx]
                    closest = (line, x_data[idx], y_data[idx])

            if closest is None or min_dist_sq > PIXEL_THRESHOLD * PIXEL_THRESHOLD:
                setattr(self, selected_attr, None)
                marker.setVisible(False)
                text_item.setVisible(False)
                return

            line, x_val, y_val = closest
            setattr(self, selected_attr, line)
            marker.setData([x_val], [y_val])
            text_item.setText(self._format_point(plot, x_val, y_val))
            self.update_marker_text_position(plot, text_item, x_val, y_val)
            marker.setVisible(True)
            text_item.setVisible(True)

        def _format_point(self, plot, x_val, y_val):
            """Label a data point, undoing the log transform on the loss plot."""
            display_y = 10**y_val if plot == self.loss_plot else y_val
            return f"x={x_val:.2f}, y={display_y:.4f}"

        def autoscale_plot(self, plot, plot_type):
            """Reset the view range for the given plot while preserving the selected-line marker."""
            if plot_type == "metrics":
                selected_attr = "selected_metrics_line"
                marker = self.marker_metrics
                marker_text = self.marker_metrics_text
            else:
                selected_attr = "selected_loss_line"
                marker = self.marker_loss
                marker_text = self.marker_loss_text

            old_selected = getattr(self, selected_attr)
            setattr(self, selected_attr, None)
            plot.enableAutoRange()
            setattr(self, selected_attr, old_selected)

            if old_selected is not None:
                x_data, y_data = old_selected.getData()
                if len(x_data) > 0:
                    x_val, y_val = x_data[-1], y_data[-1]
                    marker.setData([x_val], [y_val])
                    marker_text.setText(self._format_point(plot, x_val, y_val))
                    marker_text.setPos(x_val, y_val)
                    marker.setVisible(True)
                    marker_text.setVisible(True)

        def update_marker_text_position(
            self, plot, text_item, x_val, y_val, x_offset_frac=0.05, y_offset_frac=0.05
        ):
            """Position text_item near (x_val, y_val), nudging it inward if it would overflow the view."""
            vb = plot.getViewBox()
            (x_min, x_max), (y_min, y_max) = vb.viewRange()

            x_offset = (x_max - x_min) * x_offset_frac
            y_offset = (y_max - y_min) * y_offset_frac

            x_text = x_val
            y_text = y_val + y_offset

            margin_frac = 0.1
            margin_x = (x_max - x_min) * margin_frac
            margin_y = (y_max - y_min) * margin_frac

            if x_text + margin_x > x_max:
                x_text = x_val - x_offset
            if y_text + margin_y > y_max:
                y_text = y_val - y_offset
            if y_text - margin_y < y_min:
                y_text = y_val + y_offset

            text_item.setPos(x_text, y_text)

    app = ensure_app()
    gui = TrainingGUI(device_text=device_text)
    gui.showMaximized()

    # Filled in by the worker thread and read back on the main thread once the
    # GUI has closed, so a failure can be reported instead of leaving the
    # monitor window up with no explanation.
    outcome = {"ok": False, "error": None, "results": None}

    def train_thread():
        """Worker: load YOLO, register epoch callback, run training, then signal the GUI to close."""

        def on_epoch_end(trainer):
            """Ultralytics callback: emit per-epoch metrics to the GUI and check for stop request."""
            m = trainer.metrics

            gui.metrics_updated.emit(
                trainer.epoch,
                m.get("metrics/precision(B)", 0.0),
                m.get("metrics/recall(B)", 0.0),
                m.get("metrics/mAP50-95(B)", 0.0),
                m.get("val/box_loss", 0.0),
                m.get("val/cls_loss", 0.0),
                m.get("val/dfl_loss", 0.0),
            )

            if gui.stop_requested:
                trainer.stop = True

        try:
            model = YOLO(model_path)
            model.add_callback("on_fit_epoch_end", on_epoch_end)

            outcome["results"] = model.train(
                data=dataset_yaml,
                epochs=int(1e10),
                device=device,
                project=model_save_dir,
                name="YOLO_EZ_training",
                patience=0,
                save=True,
                plots=False,
            )
            outcome["ok"] = True
        except BaseException:
            # Without this the thread dies silently and the monitor window sits
            # there empty forever with no indication of what went wrong.
            outcome["error"] = traceback.format_exc()
            print(outcome["error"], file=sys.stderr)
        finally:
            gui.save_and_close_requested.emit()

    class _Worker(QtCore.QThread):
        """Minimal QThread subclass that executes an arbitrary callable."""

        def __init__(self, fn):
            """Args: fn: Callable to run on the worker thread."""
            super().__init__()
            self._fn = fn

        def run(self):
            """Execute the stored callable."""
            self._fn()

    thread = _Worker(train_thread)
    thread.start()
    app.exec_()
    thread.wait()

    if outcome["error"] is not None:
        # The full traceback goes to stderr; the dialog shows only the tail,
        # which is where the actual error is, so it stays readable on screen.
        tail = outcome["error"].strip().splitlines()[-12:]
        show_error_window(
            "Training stopped because of an error:\n\n" + "\n".join(tail),
            title="Training Failed",
        )
        return None

    # model.train() returns None under some configurations, so fall back to the
    # success flag; callers only need a truthy value to mean "training ran".
    return outcome["results"] if outcome["results"] is not None else outcome["ok"]
