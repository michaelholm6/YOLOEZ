import sys
import threading
import subprocess
from PyQt5 import QtWidgets, QtGui
from ultralytics import YOLO
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import webbrowser, time
import numpy as np


def run_training(model_path, dataset_yaml, tb_logdir="runs/train"):

    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Using device: {device}")

    # Ensure log directory exists
    os.makedirs(tb_logdir, exist_ok=True)

    # Launch TensorBoard
    tb_process = subprocess.Popen(
        ["uv", "run", "tensorboard", f"--logdir={tb_logdir}", "--port=6006", "--host=127.0.0.1", "--reload_interval=1"],
        stdout=None,  # show logs in console
        stderr=None
    )
    print("TensorBoard launched at http://127.0.0.1:6006")

    # GUI for stop button

    class TrainingGUI(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("YOLO Training Monitor")

            # Get screen size
            screen = QtWidgets.QApplication.primaryScreen()
            dpi = screen.logicalDotsPerInch()

            # Scale window size relative to screen
            geom = screen.availableGeometry()
            win_width = int(geom.width() * 0.25)
            win_height = int(geom.height() * 0.15)
            self.resize(win_width, win_height)

            # Center the window
            x = (geom.width() - win_width) // 2
            y = (geom.height() - win_height) // 2
            self.move(x, y)

            self.stop_requested = False

            layout = QtWidgets.QVBoxLayout(self)
            self.setLayout(layout)

            # Create button
            self.stop_button = QtWidgets.QPushButton("Stop Training")

            # Scale font with DPI
            font = QtGui.QFont()
            font.setPointSizeF(dpi / 16)  # adjust divisor for bigger/smaller
            self.stop_button.setFont(font)

            # Make button expand
            self.stop_button.setMinimumHeight(int(win_height * 0.4))

            self.stop_button.clicked.connect(self.request_stop)
            layout.addWidget(self.stop_button)

        def request_stop(self):
            self.stop_requested = True
            print("⚠ Stop requested!")
            self.close()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    gui = TrainingGUI()
    gui.show()

    # TensorBoard writer
    

# Make a unique run folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(tb_logdir, f"run_{timestamp}")
    run_name = f"run_{timestamp}"
    writer = SummaryWriter(log_dir=os.path.join(run_dir))

    def train_thread():
        model = YOLO(model_path)

        # Custom callback for per-epoch logging
        def on_epoch_end(trainer):
            epoch = trainer.epoch
            # Log training loss
            if "train/loss" in trainer.metrics:
                writer.add_scalar("Train/Loss", trainer.metrics["train/loss"], epoch)
            # Log validation mAP50
            if "metrics/mAP50(B)" in trainer.metrics:
                writer.add_scalar("Val/mAP50", trainer.metrics["metrics/mAP50(B)"], epoch)
            writer.flush()

            if gui.stop_requested:
                print("⚠ Training stopped by user.")
                trainer.stop = True  # stop training gracefully

        # Register callback
        model.add_callback("on_fit_epoch_end", on_epoch_end)

        # Run training
        model.train(
            data=dataset_yaml,
            epochs=int(10e10),
            device=device,
        )

        writer.close()
        print("Training finished.")

    thread = threading.Thread(target=train_thread, daemon=True)
    webbrowser.open('http://127.0.0.1:6006')
    thread.start()

    # Run Qt event loop
    exit_code = app.exec_()

    # Stop TensorBoard when GUI closes
    tb_process.terminate()
    tb_process.wait()
    sys.exit(exit_code)