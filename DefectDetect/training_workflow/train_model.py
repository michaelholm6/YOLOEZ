import sys
import threading
import subprocess
from PyQt5 import QtWidgets, QtGui
from ultralytics import YOLO
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import webbrowser
import shutil

def run_training(model_path, dataset_yaml, tb_logdir="runs/train"):

    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(tb_logdir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)  # make sure it exists
    writer = SummaryWriter(log_dir=run_dir)

    # Launch TensorBoard pointing ONLY at this run
    tb_process = subprocess.Popen(
    ["tensorboard", f"--logdir={run_dir}", "--port=6006", "--host=127.0.0.1", "--reload_interval=1"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)


    print("TensorBoard launched at http://127.0.0.1:6006")
    webbrowser.open('http://127.0.0.1:6006')

    # Container to store training results
    training_results = {}

    # PyQt GUI
    class TrainingGUI(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("YOLO Training Monitor")

            # Screen geometry for scaling
            screen = QtWidgets.QApplication.primaryScreen()
            dpi = screen.logicalDotsPerInch()
            geom = screen.availableGeometry()
            win_width = int(geom.width() * 0.25)
            win_height = int(geom.height() * 0.15)
            self.resize(win_width, win_height)
            self.move((geom.width() - win_width) // 2, (geom.height() - win_height) // 2)

            self.stop_requested = False

            layout = QtWidgets.QVBoxLayout(self)
            self.setLayout(layout)

            # Stop button
            self.stop_button = QtWidgets.QPushButton("Stop Training")
            font = QtGui.QFont()
            font.setPointSizeF(dpi / 8)
            self.stop_button.setFont(font)
            self.stop_button.setMinimumHeight(int(win_height * 0.4))
            self.stop_button.clicked.connect(self.request_stop)
            layout.addWidget(self.stop_button)

        def request_stop(self):
            self.stop_requested = True
            print("⚠ Stop requested by user!")
            self.close()

    # PyQt application
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    gui = TrainingGUI()
    gui.show()

    # Training thread
    def train_thread():
        model = YOLO(model_path)

        # Callback to log metrics per epoch and check for stop
        def on_epoch_end(trainer):
            epoch = trainer.epoch
            if "train/loss" in trainer.metrics:
                writer.add_scalar("Train/Loss", trainer.metrics["train/loss"], epoch)
            if "metrics/mAP50(B)" in trainer.metrics:
                writer.add_scalar("Val/mAP50", trainer.metrics["metrics/mAP50(B)"], epoch)
            writer.flush()

            if gui.stop_requested:
                print("⚠ Stopping training gracefully...")
                trainer.stop = True

        model.add_callback("on_fit_epoch_end", on_epoch_end)

        # Run training
        results = model.train(
            data=dataset_yaml,
            epochs=int(10e10),
            device=device,
        )

        writer.close()
        print("✅ Training finished.")
        training_results["results"] = results

        # Close GUI automatically if training finishes
        QtWidgets.QApplication.instance().quit()

    # Start training
    thread = threading.Thread(target=train_thread)
    thread.start()

    # Run GUI event loop (blocking)
    app.exec_()

    # Wait for training to finish
    thread.join()

    # Stop TensorBoard
    tb_process.terminate()
    tb_process.wait()
    
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    # Return YOLO Results object
    return training_results.get("results", None)
