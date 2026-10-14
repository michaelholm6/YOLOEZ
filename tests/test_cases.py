# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

"""Test suite for YOLOEZ, organised as one test per subsystem.

Each test walks a subsystem through the behaviour that matters, with a comment
marking every concern it covers.  Comments reading "regression:" mark behaviour
that was once broken; those assertions fail against the defect they describe.
"""

import math
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

import main as main_module
import training_workflow.main_training as main_training_module
import training_workflow.train_model as train_model
import YOLO_EZ
from inference_workflow.get_user_inputs_inference import InputDialogInference
from inference_workflow.inference_viewer import ImageViewer
from inference_workflow.main_inference import run_inference_workflow
from inference_workflow.run_inference import run_yolo_inference
from labelling_workflow.area_of_interest_marking import (
    PolygonAnnotatorWindow,
    PolygonCanvas,
)
from labelling_workflow.bootstrap_runner import run_yolo_on_crops
from labelling_workflow.crop_and_mask import crop_and_mask_images
from labelling_workflow.edit_box_points import (
    BoundingBoxEditorView,
    MultiImageBoxEditor,
)
from labelling_workflow.edit_contour_points import (
    ContourEditorView,
    MultiImageContourEditor,
)
from labelling_workflow.get_user_inputs_labelling import InputDialogLabelling
from labelling_workflow.main_labelling import run_labeling_workflow
from labelling_workflow.save_box_results import save_box_results
from labelling_workflow.save_segmentation_results import save_segmentation_results
from main import main as run_main
from training_workflow.apply_augmentations import (
    augment_yolo_dataset,
    detect_label_format,
)
from training_workflow.dataset_chooser import prepare_yolo_dataset, split_dataset
from training_workflow.get_user_inputs_training import YOLOTrainingDialog
from training_workflow.main_training import run_training_workflow
from utils import read_image, write_image

ASSET_IMAGES = "assets/test_images"
ASSET_LABELS = "assets/test_labels"

_qapp = None


def _ensure_qapp():
    """Offscreen QApplication shared by the tests.

    The instance is cached at module level: without a lasting Python reference it
    is garbage collected straight away and widget construction then fails.
    """
    global _qapp
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if QtWidgets.QApplication.instance() is None:
        _qapp = QtWidgets.QApplication(sys.argv)
    return QtWidgets.QApplication.instance()


def _press(widget, key, modifier=QtCore.Qt.NoModifier):
    widget.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, modifier))


def _click(widget, x, y):
    pos = QtCore.QPoint(x, y)
    widget.mousePressEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress,
            pos,
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
        )
    )
    widget.mouseReleaseEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease,
            pos,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        )
    )


def _box(x1, y1, x2, y2):
    """A box in the editor's corner order: top-left, top-right, bottom-right, bottom-left."""
    return np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)


def _square(x, y, size=40):
    return _box(x, y, x + size, y + size)


class _DummyLoop:
    """Stand-in for the QEventLoop the editors quit when the user clicks Finish."""

    def __init__(self):
        self.quit_called = False

    def quit(self):
        self.quit_called = True


def _write_dataset(root, label_body, count=4):
    """Write a flat image+label dataset and return the image paths."""
    os.makedirs(root, exist_ok=True)
    paths = []
    for i in range(count):
        image_path = os.path.join(root, f"im{i}.png")
        cv2.imwrite(image_path, np.full((40, 40, 3), 90, np.uint8))
        with open(os.path.join(root, f"im{i}.txt"), "w") as handle:
            handle.write(label_body)
        paths.append(image_path)
    return paths


def _write_split_dataset(root, label_body, image=None):
    """Write a YOLO dataset with images/labels train and val subdirectories."""
    if image is None:
        image = np.full((40, 40, 3), 200, np.uint8)
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)
        cv2.imwrite(os.path.join(root, "images", split, "img1.png"), image)
        with open(os.path.join(root, "labels", split, "img1.txt"), "w") as handle:
            handle.write(label_body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_entry_point_and_mode_dialog(monkeypatch):
    """main(), and the startup dialog's dispatch to each workflow."""
    _ensure_qapp()
    monkeypatch.setattr("sys.exit", lambda *args, **kwargs: None)

    # Test-mode short circuit used by the rest of the suite.
    assert run_main(test_mode=True) == 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)

    dispatched = []
    monkeypatch.setattr(
        YOLO_EZ, "run_labeling_workflow", lambda **kw: dispatched.append(("label", kw))
    )
    monkeypatch.setattr(
        YOLO_EZ, "run_training_workflow", lambda **kw: dispatched.append(("train", kw))
    )
    monkeypatch.setattr(
        YOLO_EZ, "run_inference_workflow", lambda **kw: dispatched.append(("use", kw))
    )

    real_init = YOLO_EZ.ChoiceDialog.__init__

    def dialog_that_acts(action):
        def init_then_act(self):
            real_init(self)
            if action == "x-close":
                QtCore.QTimer.singleShot(20, self.reject)
                return
            if action == "suppress":
                QtCore.QTimer.singleShot(10, self.checkbox.click)
            button = {
                "label": self.label_button,
                "train": self.train_button,
                "use": self.use_button,
                "suppress": self.train_button,
            }[action]
            QtCore.QTimer.singleShot(30, button.click)

        return init_then_act

    expected = [
        ("label", ("label", {"suppress_instructions": False})),
        ("train", ("train", {"suppress_instructions": False})),
        ("use", ("use", {"suppress_instructions": False})),
        ("suppress", ("train", {"suppress_instructions": True})),
    ]
    for action, want in expected:
        dispatched.clear()
        with patch.object(YOLO_EZ.ChoiceDialog, "__init__", dialog_that_acts(action)):
            YOLO_EZ.main()
        assert dispatched == [want], action

    # Closing the dialog with the X runs nothing.
    dispatched.clear()
    with patch.object(YOLO_EZ.ChoiceDialog, "__init__", dialog_that_acts("x-close")):
        YOLO_EZ.main()
    assert dispatched == []

    # regression: main() used to start a second event loop with no windows left
    # and block there forever instead of returning.
    dispatched.clear()
    with patch.object(YOLO_EZ.ChoiceDialog, "__init__", dialog_that_acts("label")):
        assert main_module.main() == 0
    assert dispatched, "the chosen workflow never ran"


# ---------------------------------------------------------------------------
# Whole workflows
# ---------------------------------------------------------------------------


@pytest.mark.timeout(180)
def test_workflows_end_to_end(monkeypatch, tmp_path):
    """Each workflow runs start to finish with its GUI steps stubbed out."""
    _ensure_qapp()

    # --- labelling ---------------------------------------------------------
    label_inputs = {
        "image_paths": [
            os.path.join(ASSET_IMAGES, f"test_image_{i}.png") for i in range(1, 6)
        ],
        "bootstrapping_model": None,
        "bootstrapping_confidence": 0.5,
        "annotation_mode": "segmentation",
        "YOLO_true": False,
        "output_folder": str(tmp_path / "labels_out"),
        "save_unlabeled_images": False,
    }
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.show_instructions", lambda **kw: None
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
        lambda images, *a, **k: {key: [] for key in images},
    )
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.run_box_editor",
        lambda images, *a, **k: {key: [] for key in images},
    )
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.run_yolo_on_crops",
        lambda images, *a, **k: {key: [] for key in images},
    )
    saved = []
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.save_segmentation_results",
        lambda *a, **k: saved.append("segmentation"),
    )
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.save_box_results",
        lambda *a, **k: saved.append("box"),
    )
    run_labeling_workflow(suppress_instructions=True, test_inputs=label_inputs)
    assert saved == ["segmentation"]

    # An empty image list is refused rather than crashing downstream.
    errors = []
    monkeypatch.setattr(
        "labelling_workflow.main_labelling.show_error_window",
        lambda message, title="Error": errors.append(title),
    )
    run_labeling_workflow(
        suppress_instructions=True, test_inputs={**label_inputs, "image_paths": []}
    )
    assert errors and saved == ["segmentation"], "the workflow ran with no images"

    # --- training ----------------------------------------------------------
    # The bundled labels are 5-field boxes, so the task has to be "detection".
    # It used to say "segment", which matches neither task, so every image was
    # skipped during augmentation and the pipeline trained on nothing.
    training_inputs = {
        "train_split": 0.8,
        "task": "detection",
        "dataset_folder": ASSET_LABELS,
        "transformations": {},
        "number_of_augs": 1,
        "save_folder": str(tmp_path / "model_out"),
        "model_size": "n",
        "prev_model_path": None,
    }
    monkeypatch.setattr(
        "training_workflow.main_training.show_instructions", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "training_workflow.main_training.run_training",
        lambda *a, **k: {"mock_result": True},
    )

    def fail_on_error(message, title="Error"):
        raise AssertionError(f"workflow reported an error: {title}: {message}")

    monkeypatch.setattr(
        "training_workflow.main_training.show_error_window", fail_on_error
    )

    # Dataset preparation and augmentation run for real here; only the training
    # itself is mocked out.
    results = run_training_workflow(
        suppress_instructions=True, test_inputs=training_inputs
    )
    assert results.get("mock_result") is True

    # regression: a task matching no label emptied the dataset and handed it to
    # YOLO, which failed with a message that said nothing about the cause.
    mismatched = tmp_path / "mismatched"
    _write_dataset(str(mismatched), "0 0.1 0.1 0.5 0.1 0.5 0.5")
    reported = []

    def refuse_to_train(*a, **k):
        raise AssertionError("training must not start on an empty dataset")

    with (
        patch.object(
            main_training_module,
            "show_error_window",
            lambda message, title="Error": reported.append((title, message)),
        ),
        patch.object(main_training_module, "run_training", refuse_to_train),
    ):
        outcome = run_training_workflow(
            suppress_instructions=True,
            test_inputs={
                **training_inputs,
                "dataset_folder": str(mismatched),
                "train_split": 0.5,
            },
        )
    assert outcome is None
    assert reported and "empty" in reported[0][1].lower()

    # --- inference ---------------------------------------------------------
    inference_inputs = {
        "image_paths": [
            os.path.join(ASSET_IMAGES, f"test_image_{i}.jpg") for i in range(1, 6)
        ],
        "YOLO_model": "fake_model.pt",
        "YOLO_confidence": 0.25,
        "output_folder": str(tmp_path / "inference_out"),
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
    monkeypatch.setattr(
        "inference_workflow.main_inference.YOLO", lambda *a, **k: object()
    )
    inferred = []

    def record_inference(*a, **k):
        inferred.append(True)
        return []

    monkeypatch.setattr(
        "inference_workflow.main_inference.run_yolo_inference", record_inference
    )
    run_inference_workflow(suppress_instructions=True, test_inputs=inference_inputs)
    assert inferred == [True]


# ---------------------------------------------------------------------------
# Input dialogs
# ---------------------------------------------------------------------------


@pytest.mark.timeout(90)
def test_input_dialogs(tmp_path):
    """The three parameter dialogs: opening, validation, and the preview list."""
    _ensure_qapp()

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(4):
        cv2.imwrite(str(image_dir / f"im{i}.png"), np.full((30, 30, 3), 40, np.uint8))
    image_files = [str(p) for p in sorted(image_dir.glob("*.png"))]

    for dialog_cls in (InputDialogLabelling, InputDialogInference, YOLOTrainingDialog):
        dialog = dialog_cls()
        dialog.show()
        assert dialog is not None
        dialog.close_flag = True
        dialog.close()

    # --- labelling: validation tracks the preview list ---------------------
    dialog = InputDialogLabelling()
    dialog.image_path_edit.setText(str(image_dir))
    dialog.output_path_edit.setText(str(tmp_path))
    dialog.bbox_radio.setChecked(True)
    dialog.image_files = list(image_files)
    dialog.current_image_index = 0
    dialog.update_run_button_state()
    assert dialog.run_button.isEnabled()

    # regression: removing every image left Run enabled, and the workflow then
    # crashed on an empty image list.
    while dialog.image_files:
        dialog.remove_current_image()
    assert not dialog.run_button.isEnabled()
    assert dialog.get_values()["image_paths"] == []
    dialog.close_flag = True
    dialog.close()

    # --- inference: an empty folder is not runnable ------------------------
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    model = tmp_path / "model.pt"
    model.write_bytes(b"not really a model")

    dialog = InputDialogInference()
    dialog.image_path_edit.setText(str(empty_dir))
    dialog.output_path_edit.setText(str(tmp_path))
    dialog.model_path_edit.setText(str(model))
    assert not dialog.run_button.isEnabled()
    dialog.close_flag = True
    dialog.close()

    # --- the preview list stays consistent through navigation and removal --
    for dialog_cls in (InputDialogLabelling, InputDialogInference):
        dialog = dialog_cls()
        with patch.object(
            QtWidgets.QFileDialog, "getExistingDirectory", return_value=str(image_dir)
        ):
            dialog.browse_image()
        assert len(dialog.image_files) == 4

        dialog.show_next_image()
        dialog.show_next_image()
        dialog.remove_current_image()  # a middle image
        dialog.show_next_image()
        dialog.remove_current_image()  # the last image

        count = len(dialog.image_files)
        index = dialog.current_image_index
        assert count == 2
        assert 0 <= index < count
        assert dialog.image_index_label.text() == f"Image {index + 1}/{count}"
        assert dialog.prev_button.isEnabled() == (index > 0)
        assert dialog.next_button.isEnabled() == (index < count - 1)
        dialog.close_flag = True
        dialog.close()

    # --- confidence threshold ---------------------------------------------
    # regression: int(0.29 * 100) truncated to 28, and the slider fed that back
    # into the spin box, silently changing the value the user typed.
    for dialog_cls in (InputDialogInference, InputDialogLabelling):
        dialog = dialog_cls()
        for wanted in (0.29, 0.57, 0.58, 0.87, 0.01, 1.0):
            dialog.confidence_spinbox.setValue(wanted)
            assert dialog.confidence_spinbox.value() == pytest.approx(
                wanted
            ), f"{dialog_cls.__name__} changed {wanted}"
            assert dialog.confidence_slider.value() == round(wanted * 100)
        dialog.close_flag = True
        dialog.close()

    # --- training dialog ---------------------------------------------------
    dataset = tmp_path / "dataset"
    _write_dataset(str(dataset), "0 0.5 0.5 0.2 0.2", count=3)
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    # regression: browse_dataset validated before populating the image list, so
    # choosing the save folder first and the dataset second left Run disabled.
    dialog = YOLOTrainingDialog()
    with patch.object(
        QtWidgets.QFileDialog, "getExistingDirectory", return_value=str(save_dir)
    ):
        dialog.browse_save()
    with patch.object(
        QtWidgets.QFileDialog, "getExistingDirectory", return_value=str(dataset)
    ):
        dialog.browse_dataset()
    assert len(dialog.image_files) == 3
    assert dialog.run_button.isEnabled(), dialog.status_label.text()
    assert dialog.task == "detection"

    # regression: "Remove Image" only changed the preview; training re-scanned
    # the folder and used the removed image anyway.
    assert len(dialog.get_values()["image_paths"]) == 3
    dialog.remove_current_image()
    assert len(dialog.get_values()["image_paths"]) == 2
    dialog.close_flag = True
    dialog.close()


# ---------------------------------------------------------------------------
# Area of interest marking
# ---------------------------------------------------------------------------


def test_area_of_interest_marking(monkeypatch):
    """Polygon drawing, the window around it, and the masking step it feeds."""
    _ensure_qapp()
    monkeypatch.setattr(sys, "exit", lambda *a, **k: None)

    canvas = PolygonCanvas()
    canvas.show()
    canvas.resize(400, 400)
    canvas.load_image(np.zeros((200, 200, 3), dtype=np.uint8))

    # regression: mousePressEvent added the point and mouseReleaseEvent added it
    # again, so every click from the third onwards duplicated a vertex.
    for x, y in [(50, 50), (150, 50), (150, 150), (50, 150)]:
        _click(canvas, x, y)
    assert len(canvas.current_polygon["points"]) == 4

    canvas.polygons = []
    canvas.current_polygon = {"points": [], "closed": False}
    for x, y in [(50, 50), (150, 50), (150, 150)]:
        _click(canvas, x, y)
    assert canvas.close_polygon() is True

    # regression: points are stored in widget coordinates and were not rescaled,
    # so resizing the window moved the region the user had outlined.
    before = canvas.get_polygon_image_coords()
    canvas.resize(800, 800)
    assert canvas.get_polygon_image_coords() == before

    # Undo and redo cover both a completed polygon and one in progress, and must
    # not raise when run past the ends of their stacks.
    canvas.undo()
    assert canvas.polygons == []
    canvas.redo()
    assert len(canvas.polygons) == 1
    for x, y in [(20, 20), (60, 20), (60, 60)]:
        _click(canvas, x, y)
    canvas.undo()
    assert canvas.current_polygon["points"] == []
    canvas.redo()
    assert len(canvas.current_polygon["points"]) == 3
    for _ in range(12):
        canvas.undo()
    for _ in range(12):
        canvas.redo()

    # --- the window -------------------------------------------------------
    # Image loading goes through utils.read_image, and an unreadable image pops
    # a modal error dialog, which cannot be exec_'d under the offscreen Qt
    # platform; patch both so a failure here reports rather than aborting.
    monkeypatch.setattr(
        "labelling_workflow.area_of_interest_marking.read_image",
        lambda path: np.zeros((100, 100, 3), np.uint8),
    )
    monkeypatch.setattr(
        "labelling_workflow.area_of_interest_marking.show_error_window",
        lambda message, title="Error": pytest.fail(f"unexpected error: {message}"),
    )
    window = PolygonAnnotatorWindow(["a.png", "b.png"], _DummyLoop())
    window.show()
    assert window.status_label.text() == "Image 1 / 2"
    window.change_image(1)
    assert window.status_label.text() == "Image 2 / 2"
    window.thickness_slider.setValue(7)
    assert window.thickness_value_label.text() == "7"
    window.finish_annotation()
    assert window.close_flag is True
    assert set(window.results) == {"a.png", "b.png"}

    # regression: an empty list indexed straight past the end of image_paths.
    with pytest.raises(ValueError):
        PolygonAnnotatorWindow([], _DummyLoop())

    # --- masking ----------------------------------------------------------
    image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    monkeypatch.setattr(
        "labelling_workflow.crop_and_mask.read_image", lambda path: image.copy()
    )
    masked = crop_and_mask_images(
        ["img1.png"], {"img1.png": [[(3, 3), (6, 3), (6, 6), (3, 6)]]}
    )["img1.png"]
    assert masked.shape == image.shape
    assert np.all(masked[4, 4] == [255, 255, 255])
    assert np.all(masked[0, 0] == [0, 0, 0])
    assert np.all(masked[9, 9] == [0, 0, 0])


# ---------------------------------------------------------------------------
# Bounding box editor
# ---------------------------------------------------------------------------


def test_box_editor(monkeypatch):
    """Selection, deletion, undo/redo, dragging, and per-image persistence."""
    _ensure_qapp()
    monkeypatch.setattr(sys, "exit", lambda *a, **k: None)

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    box = _box(50, 50, 100, 100)
    view = BoundingBoxEditorView(image, boxes=[box.copy()], line_thickness=2)
    view.show()

    # Lasso selection, then deleting whole boxes by any selected corner.
    view.select_points_in_polygon(
        QtGui.QPolygonF(
            [
                QtCore.QPointF(0, 0),
                QtCore.QPointF(150, 0),
                QtCore.QPointF(150, 150),
                QtCore.QPointF(0, 150),
            ]
        )
    )
    assert len(view.selected_points) == 4
    view.selected_points = {(0, 0)}
    view.delete_selected_boxes()
    assert view.boxes == []
    assert view.selected_points == set()

    # Undo restores the deleted box; redo removes it again.
    _press(view, QtCore.Qt.Key_Z, QtCore.Qt.ControlModifier)
    assert len(view.boxes) == 1
    _press(view, QtCore.Qt.Key_Y, QtCore.Qt.ControlModifier)
    assert view.boxes == []

    # Box creation takes two clicks, one per opposite corner.
    def press_at_scene(target, x, y):
        target.mousePressEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonPress,
                target.mapFromScene(QtCore.QPointF(x, y)),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier,
            )
        )

    def move_to_scene(target, x, y):
        target.mouseMoveEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseMove,
                target.mapFromScene(QtCore.QPointF(x, y)),
                QtCore.Qt.NoButton,
                QtCore.Qt.NoButton,
                QtCore.Qt.NoModifier,
            )
        )

    view.boxes = []
    _press(view, QtCore.Qt.Key_C)
    assert view.creating_box is True
    press_at_scene(view, 20, 20)
    assert view.boxes == [], "the first click must not commit a box"
    # The preview rectangle grows with the mouse between the two clicks; without
    # that movement the committed box would have no area.
    move_to_scene(view, 90, 90)
    press_at_scene(view, 90, 90)
    assert len(view.boxes) == 1, "the second click must commit the box"
    committed = view.boxes[0].reshape(-1, 2)
    assert committed[:, 0].max() - committed[:, 0].min() > 1
    _press(view, QtCore.Qt.Key_Escape)

    # --- dragging ---------------------------------------------------------
    def drag(selection, dx, dy, boxes):
        dragged = BoundingBoxEditorView(
            np.zeros((400, 400, 3), np.uint8),
            boxes=[b.copy() for b in boxes],
            line_thickness=2,
        )
        dragged.selected_points = set(selection)
        dragged.moving_active = True
        dragged.move_start_mouse_pos = QtCore.QPointF(0, 0)
        dragged.boxes_original_for_move = [b.copy() for b in dragged.boxes]
        dragged._apply_move_delta(dx, dy)
        return dragged

    def extents(target, index=0):
        pts = target.boxes[index].reshape(-1, 2)
        return (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )

    start = _box(100, 100, 200, 200)

    # One corner moves that corner; the opposite one stays put.
    assert extents(drag([(0, 0)], 20, 20, [start])) == (120, 120, 200, 200)
    assert extents(drag([(0, 2)], 20, 20, [start])) == (100, 100, 220, 220)
    # An edge moves only along its perpendicular axis.
    assert extents(drag([(0, 0), (0, 1)], 20, 20, [start])) == (100, 120, 200, 200)
    assert extents(drag([(0, 1), (0, 2)], 20, 20, [start])) == (100, 100, 220, 200)
    # All four corners translate the box.
    assert extents(drag([(0, i) for i in range(4)], 30, -10, [start])) == (
        130,
        90,
        230,
        190,
    )
    # Dragging a corner past its opposite one clamps instead of flipping.
    x1, y1, x2, y2 = extents(drag([(0, 0)], 500, 500, [start]))
    assert (x2 - x1, y2 - y1) == (1, 1)

    # regression: each corner was clamped on its own, so dragging a whole box
    # off the edge squashed it flat and the annotation was then dropped.
    x1, y1, x2, y2 = extents(drag([(0, i) for i in range(4)], 1000, 1000, [start]))
    assert (x2 - x1, y2 - y1) == (100, 100), f"box collapsed to {x2 - x1}x{y2 - y1}"
    assert (x2, y2) == (399, 399)

    # regression: the selection was reassigned inside the per-box loop, so with
    # one corner picked on each of two boxes only the last one kept moving.
    two = drag([(0, 0), (1, 0)], 5, 5, [_box(10, 10, 60, 60), _box(200, 200, 260, 260)])
    assert two.selected_points == {(0, 0), (1, 0)}
    two.boxes = [b.copy() for b in two.boxes_original_for_move]
    two._apply_move_delta(10, 10)
    assert extents(two, 0)[:2] == (20, 20)
    assert extents(two, 1)[:2] == (210, 210)

    # --- the multi-image shell --------------------------------------------
    images = {"a.png": image.copy(), "b.png": image.copy()}
    initial = {"a.png": [_box(10, 10, 50, 50)], "b.png": [_box(60, 60, 90, 90)]}
    shell = MultiImageBoxEditor(images, _DummyLoop(), initial_boxes=initial)
    shell.show()
    assert shell.isVisible()
    assert shell.status_label.text() == "Image 1 / 2"

    shell.editor_view.selected_points = {(0, 0)}
    shell.editor_view.delete_selected_boxes()
    assert shell.editor_view.get_edited_boxes() == []

    # regression: an empty result was treated as "never visited", so navigating
    # away and back brought the bootstrapped boxes back.
    shell.change_image(1)
    shell.change_image(-1)
    assert shell.editor_view.get_edited_boxes() == []

    # regression: the redo stack was not cleared per image, so a redo replaced
    # this image's boxes with the previous image's state.
    _press(shell.editor_view, QtCore.Qt.Key_Z, QtCore.Qt.ControlModifier)
    shell.change_image(1)
    before = len(shell.editor_view.get_edited_boxes())
    _press(shell.editor_view, QtCore.Qt.Key_Y, QtCore.Qt.ControlModifier)
    assert len(shell.editor_view.get_edited_boxes()) == before

    shell.finish_editing()
    assert shell.close_flag is True

    # regression: images the user never opened were dropped from the results,
    # losing their bootstrapped boxes.
    fresh = MultiImageBoxEditor(
        {k: v.copy() for k, v in images.items()}, _DummyLoop(), initial_boxes=initial
    )
    results = fresh.get_results()
    assert set(results) == {"a.png", "b.png"}
    assert len(results["b.png"]) == 1

    with pytest.raises(ValueError):
        MultiImageBoxEditor({}, _DummyLoop())

    # regression: a single-channel image hit a 3-channel unpack.
    grey = BoundingBoxEditorView(
        np.zeros((50, 50), dtype=np.uint8), boxes=[], line_thickness=2
    )
    assert not grey.pixmap_item.pixmap().isNull()


# ---------------------------------------------------------------------------
# Contour editor
# ---------------------------------------------------------------------------


def test_contour_editor(monkeypatch):
    """Editing operations, mode handling, and per-image persistence."""
    _ensure_qapp()
    monkeypatch.setattr(sys, "exit", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QApplication, "quit", lambda *a, **k: None)

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    first = _square(10, 10)
    second = _square(100, 100, 50)
    view = ContourEditorView(image, [first.copy(), second.copy()], 2)
    view.show()

    # Contours too small to form a polygon are dropped.
    view.contours.append(np.array([[[1, 1]]], dtype=np.int32))
    view.remove_small_contours()
    assert all(len(c) >= 3 for c in view.contours)

    # Lasso selection, then deleting the selected points.
    view.select_points_in_polygon(
        QtGui.QPolygonF(
            [
                QtCore.QPointF(0, 0),
                QtCore.QPointF(60, 0),
                QtCore.QPointF(60, 60),
                QtCore.QPointF(0, 60),
            ]
        )
    )
    assert len(view.selected_points) == 4
    view.selected_points = {(0, 0)}
    view.delete_selected_points()
    assert len(view.contours[0]) == 3

    # --- geometry ---------------------------------------------------------
    centre_x, centre_y = image.shape[1] / 2, image.shape[0] / 2

    def selected_all(contour):
        target = ContourEditorView(image, [contour.copy()], 2)
        target.selected_points = {(0, i) for i in range(len(contour))}
        return target

    # Scaling by a mouse distance ratio of 2 doubles each point's distance from
    # the contour centroid.
    scaled = selected_all(_square(60, 60))
    scaled.scaling_active = True
    scaled.contours_original_for_scaling = [c.copy() for c in scaled.contours]
    scaled.scaling_reference = {
        (0, i): tuple(scaled.contours[0][i][0]) for i in range(4)
    }
    scaled.scaling_initial_distance = 50.0
    scaled.mouse_pos = QtCore.QPointF(centre_x + 100, centre_y)
    scaled.scale_selected_points()
    assert [tuple(int(v) for v in p) for p in scaled.contours[0].reshape(-1, 2)] == [
        (40, 40),
        (120, 40),
        (120, 120),
        (40, 120),
    ]
    # Re-running with the same mouse position is not cumulative.
    unchanged = scaled.contours[0].copy()
    scaled.scale_selected_points()
    assert np.array_equal(scaled.contours[0], unchanged)

    # Rotating a square 90 degrees about its centroid maps its corners onto
    # themselves.
    rotated = selected_all(_square(60, 60))
    rotated.rotating_active = True
    rotated.contours_original_for_rotation = [c.copy() for c in rotated.contours]
    rotated.rotation_reference = {
        (0, i): tuple(rotated.contours[0][i][0]) for i in range(4)
    }
    rotated.rotation_start_angle = 0.0
    rotated.mouse_pos = QtCore.QPointF(centre_x, centre_y + 50)
    rotated.rotate_selected_points()
    assert {tuple(int(v) for v in p) for p in rotated.contours[0].reshape(-1, 2)} == {
        tuple(int(v) for v in p) for p in _square(60, 60).reshape(-1, 2)
    }

    # Moving selected points shifts them.
    moved = selected_all(_square(50, 50, 10))
    moved.moving_active = True
    moved.move_start_mouse_pos = QtCore.QPointF(0, 0)
    moved.mouse_pos = QtCore.QPointF(10, 10)
    moved.contours_original_for_move = [c.copy() for c in moved.contours]
    moved.move_selected_points()
    assert not np.array_equal(moved.contours[0], _square(50, 50, 10))

    # --- union ------------------------------------------------------------
    overlapping = ContourEditorView(image, [_square(10, 10), _square(30, 30)], 2)
    overlapping.selected_points = {(0, 0), (1, 0)}
    overlapping.union_selected_contours()
    assert len(overlapping.contours) == 1

    # regression: only the largest component was kept, so merging contours that
    # do not touch silently deleted one of them.
    disjoint = ContourEditorView(
        np.zeros((300, 300, 3), np.uint8), [_square(10, 10), _square(200, 200)], 2
    )
    disjoint.selected_points = {(0, 0), (1, 0)}
    disjoint.union_selected_contours()
    assert len(disjoint.contours) == 2

    # --- modes ------------------------------------------------------------
    modes = ContourEditorView(image, [first.copy()], 2)
    _press(modes, QtCore.Qt.Key_C)
    assert modes.creating_contour is True
    assert modes.current_mode == "Contour Creation"
    _press(modes, QtCore.Qt.Key_C)
    assert modes.creating_contour is False
    assert modes.current_mode == "Selection"

    cancellable = [
        ("scaling_active", {"contours_original_for_scaling": [first.copy()]}),
        (
            "rotating_active",
            {
                "contours_original_for_rotation": [first.copy()],
                "rotation_start_angle": 0.0,
            },
        ),
        ("moving_active", {"contours_original_for_move": [first.copy()]}),
        ("drawing", {"lasso_points": [QtCore.QPointF(1, 1)]}),
    ]
    for flag, setup in cancellable:
        setattr(modes, flag, True)
        for name, value in setup.items():
            setattr(modes, name, value)
        _press(modes, QtCore.Qt.Key_Escape)
        assert getattr(modes, flag) is False, flag

    # regression: undoing more times than points had been placed popped an empty
    # stack and killed the process.
    for placed in range(3):
        _press(modes, QtCore.Qt.Key_C)
        for i in range(placed):
            modes.new_contour_points.append((50 + i * 20, 50))
        for _ in range(placed + 2):
            _press(modes, QtCore.Qt.Key_Z, QtCore.Qt.ControlModifier)
        if modes.creating_contour:
            _press(modes, QtCore.Qt.Key_C)
    assert all(len(c) >= 3 for c in modes.contours)

    # --- the multi-image shell --------------------------------------------
    images = {
        "img1.png": np.zeros((100, 100, 3), dtype=np.uint8),
        "img2.png": np.zeros((80, 120, 3), dtype=np.uint8),
    }
    loop = _DummyLoop()
    shell = MultiImageContourEditor(images, loop, 2, detected_contours=None)
    shell.show()
    assert shell.isVisible()
    assert shell.editor_view is not None
    assert shell.index == 0
    assert shell.status_label.text() == "Image 1 / 2"
    assert shell.prev_btn.isEnabled() is False
    assert shell.next_btn.isEnabled() is True

    shell.change_image(1)
    assert shell.index == 1
    assert shell.status_label.text() == "Image 2 / 2"
    assert shell.prev_btn.isEnabled() is True
    assert shell.next_btn.isEnabled() is False

    shell.change_image(-1)
    assert shell.index == 0
    shell.change_image(0, force=True)
    assert shell.index == 0

    shell.save_current_polygon()
    assert "img1.png" in shell.results

    shell.finish_editing()
    assert shell.close_flag is True
    assert loop.quit_called is True
    close_event = QtGui.QCloseEvent()
    shell.closeEvent(close_event)
    assert close_event.isAccepted()

    # Closing with the X is a user exit; sys.exit is patched above so the
    # process survives the test.
    user_exit = MultiImageContourEditor(
        {"img.png": np.zeros((50, 50, 3), dtype=np.uint8)}, _DummyLoop()
    )
    user_exit.close_flag = False
    close_event = QtGui.QCloseEvent()
    user_exit.closeEvent(close_event)
    assert close_event.isAccepted()

    # regression: unvisited images came back as None, which crashed the save
    # step, and their bootstrapped contours were lost.
    detected = {"img2.png": [_square(20, 20)]}
    fresh = MultiImageContourEditor(
        {k: v.copy() for k, v in images.items()},
        _DummyLoop(),
        detected_contours=detected,
    )
    results = fresh.get_results()
    assert set(results) == {"img1.png", "img2.png"}
    assert all(v is not None for v in results.values())
    assert len(results["img2.png"]) == 1

    with pytest.raises(ValueError):
        MultiImageContourEditor({}, _DummyLoop())

    # regression: a single-channel image was read through a 3-byte stride.
    grey = ContourEditorView(np.zeros((50, 50), dtype=np.uint8), [], 2)
    assert not grey.pixmap_item.pixmap().isNull()


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------


def test_save_results(tmp_path):
    """Visualisations, masks and YOLO labels written by both annotation modes."""
    image_paths = ["img1.png", "img2.png", "img_missing.png"]
    masked = {
        "img1.png": np.zeros((10, 20, 3), dtype=np.uint8),
        "img2.png": np.ones((5, 5, 3), dtype=np.uint8) * 255,
    }

    # --- boxes ------------------------------------------------------------
    out = tmp_path / "boxes"
    results = save_box_results(
        image_paths,
        {"img1.png": [_box(1, 1, 5, 5), _box(0, 0, 0, 0)], "img2.png": []},
        masked,
        1,
        str(out),
        save_yolo_dataset=True,
        save_unlabeled_images=True,
    )
    assert len(results) == 2
    assert all(isinstance(img, np.ndarray) for img in results)
    assert (out / "box_visualizations" / "img1_boxes.png").exists()
    assert (out / "box_visualizations" / "img2_boxes.png").exists()
    yolo = out / "yolo_dataset"
    for name in ("img1", "img2"):
        assert (yolo / f"{name}.png").exists()
        assert (yolo / f"{name}.txt").exists()
    lines = (yolo / "img1.txt").read_text().splitlines()
    assert len(lines) == 1, "the zero-area box should not produce a label"
    parts = lines[0].split()
    assert parts[0] == "0"
    assert all(0 < float(v) < 1 for v in parts[1:])
    assert (yolo / "img2.txt").read_text().splitlines() == []
    # An image with no masked counterpart is skipped entirely.
    assert not (out / "box_visualizations" / "img_missing_boxes.png").exists()
    assert not (yolo / "img_missing.png").exists()

    # --- contours ---------------------------------------------------------
    out = tmp_path / "contours"
    results = save_segmentation_results(
        image_paths,
        {
            "img1.png": [_box(1, 1, 5, 5), np.array([[[0, 0]], [[0, 0]]], np.int32)],
            "img2.png": [],
        },
        masked,
        1,
        str(out),
        save_yolo_dataset=True,
        save_unlabeled_images=True,
    )
    assert len(results) == 2
    assert (out / "contour_visualizations" / "img1_contours.png").exists()
    assert (out / "segmentation_masks" / "img1_mask.png").exists()
    assert (out / "segmentation_masks" / "img2_mask.png").exists()
    yolo = out / "yolo_dataset"
    lines = (yolo / "img1.txt").read_text().splitlines()
    assert len(lines) == 1, "the 2-point contour should not produce a label"
    assert lines[0].split()[0] == "0"
    assert all(0 < float(v) < 1 for v in lines[0].split()[1:])
    assert (yolo / "img2.txt").read_text().splitlines() == []
    assert not (out / "segmentation_masks" / "img_missing_mask.png").exists()

    # The pipeline must work when the folders are not pure ASCII.  OpenCV's
    # imread/imwrite cannot open such a path on Windows, which silently lost the
    # whole saved dataset, so image IO goes through utils.read_image and
    # utils.write_image instead.  This asserts the end result rather than the
    # cause: where OpenCV already handles these paths it passes either way.
    # The name is built with chr() so no source escape can hide the point.
    unicode_dir = tmp_path / ("M" + chr(0x00FC) + "ller") / (chr(0x6D4B) + chr(0x8BD5))
    unicode_dir.mkdir(parents=True)
    assert not str(unicode_dir).isascii()

    source_image = np.full((60, 60, 3), 120, np.uint8)
    source_path = unicode_dir / "bild.png"
    assert write_image(str(source_path), source_image)
    assert read_image(str(source_path)) is not None

    masked = crop_and_mask_images([str(source_path)], {})
    assert len(masked) == 1, "the image could not be read back for masking"

    unicode_out = unicode_dir / "output"
    save_box_results(
        [str(source_path)],
        {str(source_path): [_box(5, 5, 40, 40)]},
        masked,
        2,
        str(unicode_out),
        True,
        False,
    )
    written = sorted(p.name for p in (unicode_out / "yolo_dataset").iterdir())
    assert written == ["bild.png", "bild.txt"], written

    # --- labels decode back to the pixels that were drawn -----------------
    width = height = 200
    square_image = {"a.png": np.zeros((height, width, 3), np.uint8)}
    for corners in [(10, 20, 110, 160), (0, 0, 199, 199), (150, 150, 199, 199)]:
        out = tmp_path / f"roundtrip_{corners[0]}_{corners[1]}"
        save_box_results(
            ["a.png"],
            {"a.png": [_box(*corners)]},
            square_image,
            2,
            str(out),
            True,
            False,
        )
        cx, cy, bw, bh = map(
            float, (out / "yolo_dataset" / "a.txt").read_text().split()[1:]
        )
        assert (cx - bw / 2) * width == pytest.approx(corners[0])
        assert (cy - bh / 2) * height == pytest.approx(corners[1])
        assert (cx + bw / 2) * width == pytest.approx(corners[2])
        assert (cy + bh / 2) * height == pytest.approx(corners[3])

    polygon = np.array([[[10, 10]], [[150, 10]], [[150, 120]], [[40, 90]]], np.int32)
    out = tmp_path / "roundtrip_polygon"
    save_segmentation_results(
        ["a.png"], {"a.png": [polygon]}, square_image, 2, str(out), True, False
    )
    values = (out / "yolo_dataset" / "a.txt").read_text().split()[1:]
    decoded = [
        (float(values[i]) * width, float(values[i + 1]) * height)
        for i in range(0, len(values), 2)
    ]
    for (dx, dy), (ox, oy) in zip(decoded, polygon.reshape(-1, 2)):
        assert dx == pytest.approx(ox)
        assert dy == pytest.approx(oy)

    # regression: the "no annotations" check ran before zero-area annotations
    # were filtered, so an empty .txt was written as a negative sample even
    # though the user asked not to save unlabeled images.
    flat = {"a.png": np.zeros((100, 100, 3), np.uint8)}
    out = tmp_path / "degenerate_boxes"
    save_box_results(
        ["a.png"], {"a.png": [_box(10, 10, 10, 10)]}, flat, 2, str(out), True, False
    )
    assert not (out / "yolo_dataset" / "a.txt").exists()

    out = tmp_path / "degenerate_contours"
    save_segmentation_results(
        ["a.png"],
        {"a.png": [np.array([[[10, 10]], [[20, 20]]], np.int32)]},
        flat,
        2,
        str(out),
        True,
        False,
    )
    assert not (out / "yolo_dataset" / "a.txt").exists()

    # regression: an editor may hand over None for an image that was never
    # opened, which used to raise straight away.
    out = tmp_path / "none_entries"
    save_segmentation_results(
        ["a.png", "b.png"],
        {"a.png": [], "b.png": None},
        {"a.png": flat["a.png"], "b.png": flat["a.png"]},
        2,
        str(out),
        True,
        True,
    )
    assert (out / "yolo_dataset" / "b.txt").exists()


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def test_dataset_preparation(tmp_path):
    """Pair collection, the train/val split, and the label-format detector."""
    source = tmp_path / "source"
    _write_dataset(str(source), "0 0.5 0.5 0.2 0.2", count=4)

    # regression: the dialog's image list was ignored, so images the user
    # removed were still trained on.
    keep = [str(source / "im0.png"), str(source / "im1.png")]
    out = tmp_path / "subset"
    assert prepare_yolo_dataset(
        str(source), str(out), train_split=0.5, image_paths=keep
    )
    copied = sorted(
        p.name
        for split in ("train", "val")
        for p in (out / "images" / split).glob("*.png")
    )
    assert copied == ["im0.png", "im1.png"]

    # Both splits always get at least one pair.
    for split in ("train", "val"):
        assert list((out / "images" / split).glob("*.png"))
        assert list((out / "labels" / split).glob("*.txt"))

    # regression: the output is flat, so identically named files from different
    # subdirectories overwrote each other.
    nested = tmp_path / "nested"
    for sub in ("a", "b"):
        sub_dir = nested / sub
        _write_dataset(str(sub_dir), "0 0.5 0.5 0.2 0.2", count=1)
        os.rename(sub_dir / "im0.png", sub_dir / "img.png")
        os.rename(sub_dir / "im0.txt", sub_dir / "img.txt")
    out = tmp_path / "nested_out"
    assert prepare_yolo_dataset(str(nested), str(out), train_split=0.5)
    names = [
        p.name
        for split in ("train", "val")
        for p in (out / "images" / split).glob("*.png")
    ]
    assert len(names) == 2, f"a file was overwritten: {names}"

    # A dataset too small to split, and a folder with nothing usable, are
    # reported rather than crashing inside scikit-learn.
    reported = []
    with patch(
        "training_workflow.dataset_chooser.show_error_window",
        lambda message, title="Error": reported.append(message),
    ):
        single = tmp_path / "single"
        _write_dataset(str(single), "0 0.5 0.5 0.2 0.2", count=1)
        assert prepare_yolo_dataset(str(single), str(tmp_path / "single_out")) is None
        assert (
            prepare_yolo_dataset(str(tmp_path / "void"), str(tmp_path / "void_out"))
            is None
        )
        # regression: an unusable folder returned a path anyway, and the caller
        # then tried to use it.
        assert split_dataset(str(tmp_path / "does_not_exist")) is None
    assert len(reported) == 3

    # regression: the temporary dataset was reused between runs, so leftovers
    # were folded into the next training set.
    stale_root = os.path.join(tempfile.gettempdir(), "yolo_dataset")
    stale = os.path.join(stale_root, "images", "train", "left_over.png")
    os.makedirs(os.path.dirname(stale), exist_ok=True)
    cv2.imwrite(stale, np.zeros((20, 20, 3), np.uint8))
    assert split_dataset(str(source), train_split=0.5) is not None
    assert not os.path.exists(stale)

    # regression: the dialog and the augmentation step used different rules, so
    # they could disagree and the image was then skipped without explanation.
    _ensure_qapp()
    dialog = YOLOTrainingDialog()
    cases = {
        "box": ("0 0.5 0.5 0.2 0.2", "detection"),
        "polygon": ("0 0.1 0.1 0.5 0.1 0.5 0.5", "segmentation"),
        "box_then_polygon": (
            """0 0.5 0.5 0.2 0.2
0 0.1 0.1 0.5 0.1 0.5 0.5""",
            "segmentation",
        ),
        "polygon_then_box": (
            """0 0.1 0.1 0.5 0.1 0.5 0.5
0 0.5 0.5 0.2 0.2""",
            "segmentation",
        ),
    }
    for name, (body, expected) in cases.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(body)
        assert detect_label_format(str(path)) == expected, name
        assert dialog.detect_label_format(str(path)) == expected, name

    empty = tmp_path / "empty_label.txt"
    empty.write_text("")
    assert detect_label_format(str(empty)) is None
    assert dialog.detect_label_format(str(empty)) == "empty"
    dialog.close_flag = True
    dialog.close()


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def test_augmentation(tmp_path):
    """Originals are copied, variants are generated, and bad lines are dropped."""
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    _write_split_dataset(
        str(dataset), "0 0.5 0.5 0.2 0.2", np.ones((10, 10, 3), np.uint8) * 255
    )

    augment_yolo_dataset(
        dataset_dir=str(dataset),
        aug_dict={
            "flip": True,
            "rotate": True,
            "scale": True,
            "color": True,
            "blur": True,
            "noise": True,
        },
        output_dir=str(output),
        num_augments=1,
        task="detection",
    )

    for split in ("train", "val"):
        images = output / "images" / split
        labels = output / "labels" / split
        assert (images / "img1.png").exists()
        assert (images / "img1_aug0.png").exists()
        assert (labels / "img1.txt").exists()
        lines = (labels / "img1_aug0.txt").read_text().splitlines()
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "0.0"
        assert all(0 <= float(v) <= 1 for v in parts[1:])

    # With nothing selected the pipeline is a no-op and labels pass through
    # unchanged, for both tasks.
    for task, body in (
        ("detection", "0 0.500000 0.500000 0.200000 0.200000"),
        ("segmentation", "0 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000"),
    ):
        root = tmp_path / f"noop_{task}"
        out = tmp_path / f"noop_{task}_out"
        _write_split_dataset(str(root), body)
        augment_yolo_dataset(str(root), {}, str(out), num_augments=1, task=task)
        assert (out / "labels" / "train" / "img1_aug0.txt").read_text().strip() == body

    # regression: a blank line raised IndexError, and an odd coordinate count
    # silently became a 2-point polygon that reads back as a bounding box.
    # The body below holds a valid polygon, a blank line, an odd coordinate
    # count and a non-numeric field.
    messy = tmp_path / "messy"
    messy_out = tmp_path / "messy_out"
    _write_split_dataset(
        str(messy),
        """0 0.1 0.1 0.3 0.1 0.3 0.3

0 0.1 0.2 0.3
0 oops 0.2 0.3 0.4 0.5 0.6
""",
        np.ones((10, 10, 3), np.uint8) * 255,
    )

    augment_yolo_dataset(
        dataset_dir=str(messy),
        aug_dict={"flip": True},
        output_dir=str(messy_out),
        num_augments=1,
        task="segmentation",
    )

    for split in ("train", "val"):
        for name in ("img1.txt", "img1_aug0.txt"):
            lines = (messy_out / "labels" / split / name).read_text().splitlines()
            assert len(lines) == 1, f"{split}/{name}: {lines}"
            assert len(lines[0].split()) == 7, lines[0]


# ---------------------------------------------------------------------------
# Training monitor
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_training_monitor(tmp_path):
    """What run_training reports, and the plots it puts on screen."""
    _ensure_qapp()

    class DummyTrainer:
        def __init__(self):
            self.metrics = {
                "metrics/precision(B)": 0.9,
                "metrics/recall(B)": 0.8,
                "metrics/mAP50-95(B)": 0.7,
                "val/box_loss": 0.1,
                "val/cls_loss": 0.2,
                "val/dfl_loss": 0.3,
            }
            self.epoch = 0
            self.stop = False

    class DummyYOLO:
        def __init__(self, model_path):
            self._callback = None

        def add_callback(self, name, func):
            self._callback = func

        def train(self, **kwargs):
            if self._callback:
                self._callback(DummyTrainer())
            return "training-results"

    captured = []

    def grab_gui(self):
        captured.append(self)

    # exec_ is patched narrowly so the event loop does not block; every real
    # widget is left alone.
    # regression: run_training returned None whatever happened, so the caller
    # could not tell success from failure.
    with (
        patch("training_workflow.train_model.YOLO", DummyYOLO),
        patch.object(QtWidgets.QApplication, "exec_", MagicMock(return_value=0)),
        patch.object(QtWidgets.QWidget, "showMaximized", grab_gui),
        patch("torch.cuda.is_available", return_value=False),
    ):
        assert (
            train_model.run_training(
                dataset_yaml="fake_dataset.yaml",
                model_save_dir=str(tmp_path),
                model_size="n",
                task="detection",
                prev_model_path=None,
            )
            == "training-results"
        )

    assert captured, "the training monitor was never shown"
    gui = captured[0]
    gui.update_metrics(0, 0.9, 0.8, 0.7, 0.1, 0.2, 0.3)
    gui.update_metrics(1, 0.95, 0.85, 0.75, 0.05, 0.15, 0.25)
    assert gui.precision_label.text() == "Precision: 0.9500"

    # showMaximized was patched out above, so give the window real geometry
    # before mapping data coordinates onto the scene.
    gui.resize(1200, 900)
    gui.show()
    QtWidgets.QApplication.processEvents()

    class FakeClick:
        def __init__(self, scene_pos):
            self._pos = scene_pos

        def scenePos(self):
            return self._pos

    def click_last_point(plot, curve):
        x_data, y_data = curve.getData()
        plot.plotItem.vb.autoRange()
        QtWidgets.QApplication.processEvents()
        gui.on_click(
            FakeClick(
                plot.plotItem.vb.mapViewToScene(QtCore.QPointF(x_data[-1], y_data[-1]))
            ),
            plot,
        )

    # regression: the plots were wired to an on_click that did not exist, so
    # clicking one did nothing and the pinned-line state stayed empty.
    click_last_point(gui.metrics_plot, gui.precision_curve)
    assert gui.selected_metrics_line is gui.precision_curve
    assert gui.marker_metrics.isVisible()

    click_last_point(gui.loss_plot, gui.box_loss_curve)
    assert gui.selected_loss_line is gui.box_loss_curve

    # The loss plot is log scaled, so its labels must undo the transform while
    # the metrics plot shows its values as they are.  Asserted on known values
    # rather than on the curve, whose display dataset pyqtgraph recomputes.
    assert "0.0500" in gui._format_point(gui.loss_plot, 1.0, math.log10(0.05))
    assert "0.9500" in gui._format_point(gui.metrics_plot, 1.0, 0.95)

    gui.closing_programmatically = True
    gui.close()

    # regression: a crash inside the worker thread left the monitor on screen
    # forever with no indication of what went wrong.
    class ExplodingYOLO:
        def __init__(self, model_path):
            pass

        def add_callback(self, name, func):
            pass

        def train(self, **kwargs):
            raise RuntimeError("dataset is unusable")

    shown = []
    with (
        patch("training_workflow.train_model.YOLO", ExplodingYOLO),
        patch.object(QtWidgets.QApplication, "exec_", MagicMock(return_value=0)),
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "training_workflow.train_model.show_error_window",
            lambda message, title="Error": shown.append((title, message)),
        ),
    ):
        assert (
            train_model.run_training(
                dataset_yaml="fake_dataset.yaml",
                model_save_dir=str(tmp_path),
                model_size="n",
                task="detection",
                prev_model_path=None,
            )
            is None
        )
    assert shown, "the user was never told that training failed"
    assert "dataset is unusable" in shown[0][1]

    # A missing previous model is refused before anything starts.
    shown.clear()
    with patch(
        "training_workflow.train_model.show_error_window",
        lambda message, title="Error": shown.append((title, message)),
    ):
        assert (
            train_model.run_training(
                dataset_yaml="fake_dataset.yaml",
                model_save_dir=str(tmp_path),
                model_size="n",
                task="detection",
                prev_model_path=str(tmp_path / "not_a_real_model.pt"),
            )
            is None
        )
    assert shown


# ---------------------------------------------------------------------------
# Inference and bootstrapping
# ---------------------------------------------------------------------------


def test_inference_and_bootstrapping(tmp_path, monkeypatch):
    """Detections written to disk, and the model-assisted pre-annotation step."""
    _ensure_qapp()
    import json

    import torch
    from ultralytics.engine.results import Boxes, Masks

    height, width = 400, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # A mask at the model's own resolution, not the image's.
    mask = np.zeros((1, 100, 100), dtype=np.float32)
    mask[0, 25:75, 25:75] = 1.0

    class FakeResult:
        def __init__(self):
            self.boxes = Boxes(
                torch.tensor([[100.0, 100.0, 300.0, 300.0, 0.9, 0.0]]),
                (height, width),
            )
            self.masks = Masks(torch.from_numpy(mask), (height, width))
            self.names = {0: "defect"}

        def plot(self, **kwargs):
            return image.copy()

    saved = run_yolo_inference(
        lambda _img, **kw: [FakeResult()], {"a.png": image}, str(tmp_path), conf=0.5
    )
    assert len(saved) == 1
    assert os.path.exists(saved[0])

    detections = json.loads((tmp_path / "result_1.json").read_text())
    assert detections[0]["class_name"] == "defect"
    assert detections[0]["confidence"] == pytest.approx(0.9, abs=1e-3)

    # regression: the polygon was taken from masks.data, which is at the model's
    # letterboxed resolution, so it did not line up with the bbox beside it.
    polygon = np.array(detections[0]["segmentation"], dtype=float)
    bbox = detections[0]["bbox"]
    assert polygon[:, 0].max() > width / 2, (
        f"polygon x max {polygon[:, 0].max()} looks like mask coordinates "
        f"rather than image coordinates (bbox is {bbox})"
    )
    assert polygon[:, 0].min() == pytest.approx(bbox[0], abs=width * 0.1)
    assert polygon[:, 1].min() == pytest.approx(bbox[1], abs=height * 0.1)

    # Nothing to do is not an error.
    assert run_yolo_inference(lambda *a, **k: [], {}, str(tmp_path), conf=0.5) == []

    # regression: an empty result list indexed past the end of image_paths.
    with pytest.raises(ValueError):
        ImageViewer([])

    viewer = ImageViewer(saved, loop=_DummyLoop())
    assert viewer.image_index_label.text() == "Image 1 / 1"
    viewer.finish()
    assert viewer.close_flag is True

    # --- bootstrapping ----------------------------------------------------
    class DummyBoxes:
        def __init__(self):
            self.xyxy = [np.array([10, 20, 30, 40]), np.array([50, 60, 70, 80])]
            self.conf = [0.9, 0.8]

        def __len__(self):
            return len(self.xyxy)

    class DummyBoxResult:
        boxes = DummyBoxes()
        masks = None

    class DummyBoxYOLO:
        def __init__(self, *a, **k):
            class DummyModel:
                model_type = "bounding_box"

            self.model = DummyModel()

        def predict(self, img, **kwargs):
            return [DummyBoxResult()]

    monkeypatch.setattr("labelling_workflow.bootstrap_runner.YOLO", DummyBoxYOLO)
    images = {
        "img1.png": np.zeros((100, 100, 3), dtype=np.uint8),
        "img2.png": np.zeros((200, 200, 3), dtype=np.uint8),
    }
    results = run_yolo_on_crops(
        images,
        model_path="fake_model.pt",
        confidence_threshold=0.5,
        annotation_mode="bounding_box",
    )
    assert set(results) == set(images)
    for contours in results.values():
        assert len(contours) == 2
        for contour in contours:
            assert contour.shape == (4, 1, 2)

    # Segmentation masks are scaled to the original image before contours are
    # taken, so bootstrapped contours land on the object: the blob below covers
    # 50-75% of the width and 25-50% of the height of a 600x400 image.
    seg_mask = np.zeros((1, 160, 160), np.float32)
    seg_mask[0, 40:80, 80:120] = 1.0

    class DummyConf:
        def cpu(self):
            return self

        def numpy(self):
            return np.array([0.9], np.float32)

    class DummyMasks:
        data = torch.from_numpy(seg_mask)

    class DummySegBoxes:
        conf = DummyConf()

    class DummySegResult:
        masks = DummyMasks()
        boxes = DummySegBoxes()
        orig_shape = (400, 600)

    class DummySegYOLO:
        def __init__(self, *a, **k):
            self.model = type("DummyModel", (), {})()

        def predict(self, img, **kwargs):
            return [DummySegResult()]

    monkeypatch.setattr("labelling_workflow.bootstrap_runner.YOLO", DummySegYOLO)
    monkeypatch.setattr(
        "labelling_workflow.bootstrap_runner.show_error_window", lambda *a, **k: None
    )
    seg_results = run_yolo_on_crops(
        {"a.png": np.zeros((400, 600, 3), np.uint8)},
        model_path="fake_model.pt",
        confidence_threshold=0.3,
        annotation_mode="segmentation",
    )
    points = seg_results["a.png"][0].reshape(-1, 2)
    assert 280 <= points[:, 0].min() <= 320
    assert 430 <= points[:, 0].max() <= 470
    assert 90 <= points[:, 1].min() <= 110
    assert 190 <= points[:, 1].max() <= 210
