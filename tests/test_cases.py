import pytest
from main import main as run_main


def test_main_runs_without_error(monkeypatch):
    monkeypatch.setattr("sys.exit", lambda *args, **kwargs: None)

    run_main(test_mode=True)

    assert True


from labelling_workflow.main_labelling import run_labeling_workflow


def test_labeling_workflow_runs_without_gui(tmp_path, monkeypatch):

    test_inputs = {
        "image_paths": [
            "assets/test_images/test_image_1.png",
            "assets/test_images/test_image_2.png",
            "assets/test_images/test_image_3.png",
            "assets/test_images/test_image_4.png",
            "assets/test_images/test_image_5.png",
        ],
        "bootstrapping_model": None,
        "bootstrapping_confidence": 0.5,
        "annotation_mode": "segmentation",
        "YOLO_true": False,
        "output_folder": str(tmp_path),
        "save_unlabeled_images": False,
    }

    monkeypatch.setattr(
        "labelling_workflow.main_labelling.show_instructions", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.annotate_images", lambda paths: ({}, 1)
    )
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.crop_and_mask_images",
        lambda images, areas: {p: None for p in images},
    )
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.run_contour_editor",
        lambda images, *a, **k: {k: [] for k in images},
    )
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.run_box_editor",
        lambda *args, **kwargs: [{}] * len(test_inputs["image_paths"]),
    )
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.run_yolo_on_crops",
        lambda images, *args, **kwargs: {k: [] for k in images},
    )
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.save_segmentation_results",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.save_box_results",
        lambda *args, **kwargs: None,
    )

    run_labeling_workflow(suppress_instructions=True, test_inputs=test_inputs)

    assert True


from training_workflow.main_training import run_training_workflow


def test_training_workflow_runs_without_gui(monkeypatch, tmp_path):

    test_inputs = {
        "train_split": 0.8,
        "task": "segment",
        "dataset_folder": "assets/test_labels",
        "transformations": [],
        "number_of_augs": 1,
        "save_folder": tmp_path,
        "model_size": "nano",
        "prev_model_path": None,
    }

    # Patch actual module functions, not local imports
    monkeypatch.setattr(
        "training_workflow.main_training.show_instructions",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "training_workflow.apply_augmentations.augment_yolo_dataset",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "training_workflow.create_yaml.create_yaml",
        lambda output_dir: str(tmp_path / "dataset.yaml"),
    )
    monkeypatch.setattr(
        "training_workflow.main_training.run_training",
        lambda *args, **kwargs: {"mock_result": True},
    )
    monkeypatch.setattr(
        "training_workflow.dataset_chooser.split_dataset", lambda path, **kwargs: path
    )

    results = run_training_workflow(suppress_instructions=True, test_inputs=test_inputs)

    assert results.get("mock_result") is True


from inference_workflow.main_inference import run_inference_workflow


def test_inference_workflow_runs_without_gui(monkeypatch, tmp_path):

    test_inputs = {
        "image_paths": [
            "assets/test_images/test_image_1.jpg",
            "assets/test_images/test_image_2.jpg",
            "assets/test_images/test_image_3.jpg",
            "assets/test_images/test_image_4.jpg",
            "assets/test_images/test_image_5.jpg",
        ],
        "YOLO_model": "fake_model.pt",
        "YOLO_confidence": 0.25,
        "output_folder": tmp_path,
    }

    monkeypatch.setattr(
        "inference_workflow.main_inference.show_instructions", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "inference_workflow.main_inference.annotate_images",
        lambda images: ({img: None for img in images}, 1),
    )
    monkeypatch.setattr(
        "inference_workflow.main_inference.crop_and_mask_images",
        lambda images, aois: images,
    )

    class DummyYOLO:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("inference_workflow.main_inference.YOLO", DummyYOLO)
    monkeypatch.setattr(
        "inference_workflow.main_inference.run_yolo_inference", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "os.makedirs", lambda path, exist_ok=True: None
    )  # prevent folder creation

    result = run_inference_workflow(suppress_instructions=True, test_inputs=test_inputs)

    assert True


from training_workflow.get_user_inputs_training import YOLOTrainingDialog

from unittest.mock import patch


@pytest.mark.timeout(10)
def test_training_input_dialog_can_open_and_close(qtbot):
    dlg = YOLOTrainingDialog()
    qtbot.addWidget(dlg)

    # Show the dialog
    dlg.show()
    qtbot.wait(100)

    # Fill in required fields so Run button is enabled
    dlg.dataset_path_edit.setText(".")  # existing folder
    dlg.save_path_edit.setText(".")  # existing folder
    dlg.update_run_button_state()

    assert dlg is not None

    dlg.close_flag = True
    dlg.close()


import os
from PyQt5 import QtWidgets, QtCore

from labelling_workflow.get_user_inputs_labelling import InputDialogLabelling


def test_labelling_input_dialog_can_open_and_close():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dialog = InputDialogLabelling()
    dialog.show()

    assert dialog is not None

    dialog.close_flag = True
    dialog.close()


from inference_workflow.get_user_inputs_inference import InputDialogInference


def test_inference_input_dialog_can_open_and_close():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    dialog = InputDialogInference()
    dialog.show()

    assert dialog is not None

    dialog.close_flag = True
    dialog.close()


import tempfile
import cv2
import numpy as np
from labelling_workflow.image_gatherer import load_images_from_folder


def test_load_images_from_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        img1_path = os.path.join(tmpdir, "image1.jpg")
        img2_path = os.path.join(tmpdir, "image2.png")
        img3_path = os.path.join(tmpdir, "not_an_image.txt")

        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)

        cv2.imwrite(img1_path, dummy_img)
        cv2.imwrite(img2_path, dummy_img)

        with open(img3_path, "w") as f:
            f.write("this is not an image")

        # Call function
        images = load_images_from_folder(tmpdir)

        # Assertions
        assert len(images) == 2, "Should load exactly 2 image files"
        for img in images:
            assert isinstance(img, np.ndarray), "Loaded item should be a numpy array"
            assert img.shape == (10, 10, 3), "Image shape should match dummy image"
