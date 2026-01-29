import pytest
from main import main as run_main
import sys


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


from labelling_workflow.area_of_interest_marking import PolygonAnnotatorWindow
import numpy as np


def test_polygon_annotator_window_can_open_and_close(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    monkeypatch.setattr(sys, "exit", lambda *a, **k: None)

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr("cv2.imread", lambda path: dummy_img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    image_paths = ["fake1.png", "fake2.png"]

    class DummyLoop:
        def quit(self):
            pass

    loop = DummyLoop()

    win = PolygonAnnotatorWindow(image_paths=image_paths, loop=loop)
    win.show()

    assert win is not None

    win.close_flag = True
    win.close()


from labelling_workflow.bootstrap_runner import run_yolo_on_crops


def test_run_yolo_on_crops_bounding_box(monkeypatch):
    class DummyBoxes:
        def __init__(self):
            self.xyxy = [
                np.array([10, 20, 30, 40]),
                np.array([50, 60, 70, 80]),
            ]
            self.conf = [0.9, 0.8]

        def __len__(self):
            return len(self.xyxy)

    class DummyResult:
        def __init__(self):
            self.boxes = DummyBoxes()
            self.masks = None

    class DummyYOLO:
        def __init__(self, *args, **kwargs):
            class DummyModel:
                model_type = "bounding_box"

            self.model = DummyModel()

        def predict(self, img, verbose=False, conf=0.5, save=False):
            return [DummyResult()]

    monkeypatch.setattr(
        "labelling_workflow.bootstrap_runner.YOLO",
        DummyYOLO,
    )

    images_dict = {
        "img1.png": np.zeros((100, 100, 3), dtype=np.uint8),
        "img2.png": np.zeros((200, 200, 3), dtype=np.uint8),
    }

    results = run_yolo_on_crops(
        images_dict,
        model_path="fake_model.pt",
        confidence_threshold=0.5,
        annotation_mode="bounding_box",
    )

    assert isinstance(results, dict)
    assert set(results.keys()) == set(images_dict.keys())

    for contours in results.values():
        assert len(contours) == 2
        for contour in contours:
            assert contour.shape == (4, 1, 2)


from labelling_workflow.crop_and_mask import crop_and_mask_images


def test_crop_and_mask_images_masks_outside_aoi(monkeypatch):
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255

    def fake_imread(path):
        return img.copy()

    monkeypatch.setattr("cv2.imread", fake_imread)

    image_paths = ["img1.png"]

    aois_dict = {"img1.png": [[(3, 3), (6, 3), (6, 6), (3, 6)]]}

    results = crop_and_mask_images(image_paths, aois_dict)

    assert "img1.png" in results
    masked = results["img1.png"]

    assert masked.shape == img.shape

    assert np.all(masked[4, 4] == [255, 255, 255])

    assert np.all(masked[0, 0] == [0, 0, 0])
    assert np.all(masked[9, 9] == [0, 0, 0])


from labelling_workflow.edit_box_points import MultiImageBoxEditor


def test_multi_image_box_editor_can_open_and_close(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(sys, "exit", lambda *a, **k: None)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    images = {
        "img1": np.zeros((100, 100, 3), dtype=np.uint8),
        "img2": np.zeros((120, 80, 3), dtype=np.uint8),
    }

    class DummyLoop:
        def quit(self):
            pass

    loop = DummyLoop()
    win = MultiImageBoxEditor(images, loop)
    win.show()

    assert win.isVisible()

    win.finish_editing()

    assert win.close_flag is True

    win.close()


from labelling_workflow.edit_contour_points import MultiImageContourEditor


def test_multi_image_contour_editor_can_open_and_close(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    monkeypatch.setattr(sys, "exit", lambda *a, **k: None)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    images = {
        "img1.png": np.zeros((100, 100, 3), dtype=np.uint8),
        "img2.png": np.zeros((80, 120, 3), dtype=np.uint8),
    }

    class DummyLoop:
        def quit(self):
            pass

    loop = DummyLoop()

    win = MultiImageContourEditor(
        image_dict=images,
        loop=loop,
        line_thickness=2,
        detected_contours=None,
    )

    win.show()

    assert win.isVisible()
    assert win.editor_view is not None

    win.finish_editing()

    assert win.close_flag is True


import numpy as np
from labelling_workflow.save_box_results import save_box_results


def test_save_box_results_creates_files_and_returns_images(tmp_path):
    image_paths = ["img1.png", "img2.png", "img_missing.png"]

    masked_images_dict = {
        "img1.png": np.zeros((10, 20, 3), dtype=np.uint8),
        "img2.png": np.ones((5, 5, 3), dtype=np.uint8) * 255,
    }

    boxes_dict = {
        "img1.png": [
            np.array([[[1, 1]], [[5, 1]], [[5, 5]], [[1, 5]]]),
            np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]]),
        ],
        "img2.png": [],
    }

    line_thickness = 1
    output_dir = tmp_path / "output"

    results = save_box_results(
        image_paths,
        boxes_dict,
        masked_images_dict,
        line_thickness,
        str(output_dir),
        save_yolo_dataset=True,
        save_unlabeled_images=True,
    )

    assert len(results) == 2
    assert all(isinstance(img, np.ndarray) for img in results)

    box_vis_dir = output_dir / "box_visualizations"
    assert (box_vis_dir / "img1_boxes.png").exists()
    assert (box_vis_dir / "img2_boxes.png").exists()

    yolo_dir = output_dir / "yolo_dataset"
    for img_name in ["img1", "img2"]:
        assert (yolo_dir / f"{img_name}.png").exists()
        assert (yolo_dir / f"{img_name}.txt").exists()

    txt_path = yolo_dir / "img1.txt"
    with open(txt_path) as f:
        lines = f.read().splitlines()
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    cx, cy, bw, bh = map(float, parts[1:])
    assert 0 < bw < 1 and 0 < bh < 1
    assert 0 < cx < 1 and 0 < cy < 1

    txt_path2 = yolo_dir / "img2.txt"
    with open(txt_path2) as f:
        lines2 = f.read().splitlines()
    assert lines2 == []

    assert not (box_vis_dir / "img_missing_boxes.png").exists()
    assert not (yolo_dir / "img_missing.png").exists()


from labelling_workflow.save_segmentation_results import save_segmentation_results


def test_save_segmentation_results_creates_files_and_returns_images(tmp_path):
    image_paths = ["img1.png", "img2.png", "img_missing.png"]

    masked_images_dict = {
        "img1.png": np.zeros((10, 20, 3), dtype=np.uint8),
        "img2.png": np.ones((5, 5, 3), dtype=np.uint8) * 255,
    }

    contours_dict = {
        "img1.png": [
            np.array([[[1, 1]], [[5, 1]], [[5, 5]], [[1, 5]]]),
            np.array([[[0, 0]], [[0, 0]]]),
        ],
        "img2.png": [],
    }

    line_thickness = 1
    output_dir = tmp_path / "output"

    results = save_segmentation_results(
        image_paths,
        contours_dict,
        masked_images_dict,
        line_thickness,
        str(output_dir),
        save_yolo_dataset=True,
        save_unlabeled_images=True,
    )

    assert len(results) == 2
    assert all(isinstance(img, np.ndarray) for img in results)

    contour_vis_dir = output_dir / "contour_visualizations"
    assert (contour_vis_dir / "img1_contours.png").exists()
    assert (contour_vis_dir / "img2_contours.png").exists()

    mask_dir = output_dir / "segmentation_masks"
    assert (mask_dir / "img1_mask.png").exists()
    assert (mask_dir / "img2_mask.png").exists()

    yolo_dir = output_dir / "yolo_dataset"
    for img_name in ["img1", "img2"]:
        assert (yolo_dir / f"{img_name}.png").exists()
        assert (yolo_dir / f"{img_name}.txt").exists()
    txt_path = yolo_dir / "img1.txt"
    with open(txt_path) as f:
        lines = f.read().splitlines()
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    coords = list(map(float, parts[1:]))
    assert all(0 < p < 1 for p in coords)

    txt_path2 = yolo_dir / "img2.txt"
    with open(txt_path2) as f:
        lines2 = f.read().splitlines()
    assert lines2 == []

    assert not (contour_vis_dir / "img_missing_contours.png").exists()
    assert not (mask_dir / "img_missing_mask.png").exists()
    assert not (yolo_dir / "img_missing.png").exists()


import cv2
from training_workflow.apply_augmentations import augment_yolo_dataset


def test_augment_yolo_dataset_creates_files(tmp_path):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"

    for split in ["train", "val"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        img_path = img_dir / "img1.png"
        cv2.imwrite(str(img_path), img)

        lbl_path = lbl_dir / "img1.txt"
        with open(lbl_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

    aug_dict = {
        "flip": True,
        "rotate": True,
        "scale": True,
        "color": True,
        "blur": True,
        "noise": True,
    }

    augment_yolo_dataset(
        dataset_dir=str(dataset_dir),
        aug_dict=aug_dict,
        output_dir=str(output_dir),
        num_augments=1,
        task="detection",
    )

    for split in ["train", "val"]:
        img_out_dir = output_dir / "images" / split
        lbl_out_dir = output_dir / "labels" / split

        orig_img_path = img_out_dir / "img1.png"
        aug_img_path = img_out_dir / "img1_aug0.png"
        assert orig_img_path.exists()
        assert aug_img_path.exists()

        orig_lbl_path = lbl_out_dir / "img1.txt"
        aug_lbl_path = lbl_out_dir / "img1_aug0.txt"
        assert orig_lbl_path.exists()
        assert aug_lbl_path.exists()

        with open(aug_lbl_path) as f:
            lines = f.read().splitlines()
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "0.0"
        coords = list(map(float, parts[1:]))
        assert all(0 <= c <= 1 for c in coords)
