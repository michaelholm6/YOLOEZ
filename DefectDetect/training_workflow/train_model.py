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

def run_training(dataset_yaml, model_save_dir, model_size, task="detection", prev_model_path=None):

    if task == "segmentation":
        default_model_path = f"yolo11{model_size}-seg.pt"
    else:
        default_model_path = f"yolo12{model_size}.pt"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Using device: {device}")

    # --- Handle previous model ---
    if prev_model_path:
        if not os.path.isfile(prev_model_path):
            raise FileNotFoundError(f"Provided previous model path does not exist: {prev_model_path}")
        model_path = prev_model_path
        print(f"✅ Continuing training from previous model: {model_path}")
    else:
        model_path = default_model_path
        print(f"ℹ Starting training from default weights: {model_path}")


    # ---------------- GUI ----------------
    class TrainingGUI(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("YOLO Training Monitor")

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

            # ===== Metric Labels =====
            metrics_layout = QtWidgets.QVBoxLayout()
            self.precision_container, self.precision_label = make_label_with_tooltip(
                "Precision: 0.0",
                "Precision = How reliable the model is when it says an object is present."
            )
            self.recall_container, self.recall_label = make_label_with_tooltip(
                "Recall: 0.0",
                "Recall = How well the model finds all objects present."
            )
            self.map_container, self.map_label = make_label_with_tooltip(
                "mAP0.5–0.95: 0.0",
                "How good the model is at locating and classifying objects."
            )
            for c in [self.precision_container, self.recall_container, self.map_container]:
                metrics_layout.addWidget(c)
            layout.addLayout(metrics_layout)

            # ===== Metrics Plot =====
            self.metrics_plot = pg.PlotWidget(title="Metrics")
            self.metrics_plot.addLegend()
            self.metrics_plot.setLabel("left", "Value")
            self.metrics_plot.setLabel("bottom", "Epoch")
            layout.addWidget(self.metrics_plot)

            self.precision_curve = self.metrics_plot.plot(pen="r", name="Precision")
            self.recall_curve    = self.metrics_plot.plot(pen="b", name="Recall")
            self.map_curve       = self.metrics_plot.plot(pen="g", name="mAP0.5–0.95")

            # ===== Loss Labels =====
            loss_labels_layout = QtWidgets.QVBoxLayout()
            self.box_loss_container, self.box_loss_label = make_label_with_tooltip(
                "Val Box Loss: 0.0",
                "Bounding box regression loss."
            )
            self.cls_loss_container, self.cls_loss_label = make_label_with_tooltip(
                "Val Cls Loss: 0.0",
                "Classification loss."
            )
            self.dfl_loss_container, self.dfl_loss_label = make_label_with_tooltip(
                "Val DFL Loss: 0.0",
                "Distribution Focal Loss."
            )
            for c in [self.box_loss_container, self.cls_loss_container, self.dfl_loss_container]:
                loss_labels_layout.addWidget(c)
            layout.addLayout(loss_labels_layout)

            # ===== Loss Plot =====
            self.loss_plot = pg.PlotWidget(title="Validation Losses")
            self.loss_plot.addLegend()
            self.loss_plot.setLabel("left", "Loss (log scale)")
            self.loss_plot.setLabel("bottom", "Epoch")
            self.loss_plot.setLogMode(y=True)
            layout.addWidget(self.loss_plot)

            self.box_loss_curve = self.loss_plot.plot(pen="y", name="Box Loss")
            self.cls_loss_curve = self.loss_plot.plot(pen="c", name="Cls Loss")
            self.dfl_loss_curve = self.loss_plot.plot(pen="m", name="DFL Loss")

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
            self.marker_metrics = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('y'))
            self.marker_metrics_text = pg.TextItem(anchor=(0,1), color='y')
            self.metrics_plot.addItem(self.marker_metrics)
            self.metrics_plot.addItem(self.marker_metrics_text)
            self.marker_metrics.setVisible(False)
            self.marker_metrics_text.setVisible(False)

            self.marker_loss = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('y'))
            self.marker_loss_text = pg.TextItem(anchor=(0,1), color='y')
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
            self.box_loss_label.setText(f"Val Box Loss: {box_loss:.4f}")
            self.cls_loss_label.setText(f"Val Cls Loss: {cls_loss:.4f}")
            self.dfl_loss_label.setText(f"Val DFL Loss: {dfl_loss:.4f}")

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

            loss_exporter = ImageExporter(self.loss_plot.plotItem)
            loss_file = os.path.join(plots_dir, f"loss_plot_{timestamp}.png")
            loss_exporter.export(loss_file)
            print(f"✅ Saved loss plot: {loss_file}")

        def on_click(self, evt, plot):
            pos = evt.scenePos()
            if not plot.sceneBoundingRect().contains(pos):
                return
            mouse_point = plot.plotItem.vb.mapSceneToView(pos)
            x_mouse, y_mouse = mouse_point.x(), mouse_point.y()

            if plot == self.metrics_plot:
                lines = [self.precision_curve, self.recall_curve, self.map_curve]
                marker, text_item, selected_attr = self.marker_metrics, self.marker_metrics_text, 'selected_metrics_line'
            else:
                lines = [self.box_loss_curve, self.cls_loss_curve, self.dfl_loss_curve]
                marker, text_item, selected_attr = self.marker_loss, self.marker_loss_text, 'selected_loss_line'

            current_selected = getattr(self, selected_attr)

            for line in lines:
                x_data, y_data = line.getData()
                if len(x_data) == 0:
                    continue
                idx = np.argmin(np.abs(x_data - x_mouse))
                y_range = max(y_data) - min(y_data)
                threshold = max(y_range * 0.05, 0.01)
                if abs(y_data[idx] - y_mouse) < threshold:
                    if current_selected == line:
                        setattr(self, selected_attr, None)
                        marker.setVisible(False)
                        text_item.setVisible(False)
                    else:
                        setattr(self, selected_attr, line)
                        marker.setVisible(True)
                        text_item.setVisible(True)
                    return

        def on_mouse_move(self, evt, plot):
            if plot == self.metrics_plot:
                selected_line = self.selected_metrics_line
                marker, text_item = self.marker_metrics, self.marker_metrics_text
            else:
                selected_line = self.selected_loss_line
                marker, text_item = self.marker_loss, self.marker_loss_text

            if selected_line is None:
                return

            pos = evt
            if isinstance(evt, tuple):
                pos = evt[0]
            if not plot.sceneBoundingRect().contains(pos):
                return

            mouse_point = plot.plotItem.vb.mapSceneToView(pos)
            x_mouse = mouse_point.x()

            x_data, y_data = selected_line.getData()
            if len(x_data) == 0:
                return
            
            idx = np.argmin(np.abs(x_data - x_mouse))
            x_val, y_val = x_data[idx], 10**(y_data[idx]) if plot == self.loss_plot else y_data[idx]

            marker.setData([x_val], [math.log10(y_val)] if plot == self.loss_plot else [y_val])
            text_item.setText(f"x={x_val:.2f}, y={y_val:.4f}")
            text_item.setPos(x_val, math.log10(y_val) if plot == self.loss_plot else y_val)

        def autoscale_plot(self, plot, plot_type):
            if plot_type == 'metrics':
                selected_attr, marker, marker_text = 'selected_metrics_line', self.marker_metrics, self.marker_metrics_text
            else:
                selected_attr, marker, marker_text = 'selected_loss_line', self.marker_loss, self.marker_loss_text

            old_selected = getattr(self, selected_attr)
            setattr(self, selected_attr, None)  # temporarily disable marker updates
            plot.enableAutoRange()
            setattr(self, selected_attr, old_selected)  # restore selection

            if old_selected is not None:
                x_data, y_data = old_selected.getData()
                if len(x_data) > 0:
                    x_val, y_val = x_data[-1], y_data[-1]
                    marker.setData([x_val], [y_val])
                    marker_text.setText(f"x={x_val:.2f}, y={y_val:.4f}")
                    marker_text.setPos(x_val, y_val)


    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    gui = TrainingGUI()
    gui.show()

    def train_thread():
        model = YOLO(model_path)

        def on_epoch_end(trainer):
            m = trainer.metrics

            gui.update_metrics(
                epoch=trainer.epoch,
                precision=m.get("metrics/precision(B)", 0.0),
                recall=m.get("metrics/recall(B)", 0.0),
                map_value=m.get("metrics/mAP(B)", 0.0),
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
            name="weights",
            patience=0,
            save=True
        )
        
        gui.save_plots(model_save_dir)

        gui.close()

    thread = threading.Thread(target=train_thread)
    thread.start()
    app.exec_()
    thread.join()
