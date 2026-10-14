# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import os
from utils import show_error_window, show_instructions
from training_workflow.apply_augmentations import augment_yolo_dataset
from training_workflow.create_yaml import create_yaml
from training_workflow.train_model import run_training
from training_workflow.get_user_inputs_training import get_training_inputs
from training_workflow.dataset_chooser import split_dataset
import shutil
import tempfile


def run_training_workflow(suppress_instructions=False, test_inputs=None):
    """Execute the full training pipeline: collect inputs, split data, augment, create YAML, train, and clean up.

    Args:
        suppress_instructions: Skip all instructional popups when True.
        test_inputs: If provided, bypasses the user input dialog and uses this dict directly.
    """
    if not suppress_instructions:
        show_instructions(
            "Welcome to the Training Workflow!\n\n"
            "First, you will provide inputs to set up your training session.\n"
            "You will select your dataset, training parameters, and augmentation options.\n\n"
            "After that, the training process will begin automatically.\n\n"
            "Close a window at any time to exit the program."
        )

    if test_inputs is not None:
        inputs = test_inputs
    else:
        inputs = get_training_inputs()

    if not inputs:
        return None

    split = inputs["train_split"]
    task = inputs["task"]
    dataset_path = inputs["dataset_folder"]
    augs = inputs["transformations"]
    number_of_augs = inputs["number_of_augs"]
    model_save_dir = inputs["save_folder"]
    model_size = inputs["model_size"]
    previous_model_path = inputs.get("prev_model_path", None)
    image_paths = inputs.get("image_paths")

    dataset_path = split_dataset(
        dataset_path, train_split=split, image_paths=image_paths
    )
    if dataset_path is None:
        # split_dataset has already told the user what went wrong.
        return None

    augmented_path = os.path.join(tempfile.gettempdir(), "augmented_dataset")
    # Clear it first so augmented images from an earlier run are not trained on.
    shutil.rmtree(augmented_path, ignore_errors=True)

    augment_yolo_dataset(
        dataset_path,
        augs,
        augmented_path,
        number_of_augs,
        task=task,
    )

    # Images whose labels do not match the task are skipped during augmentation.
    # If that leaves either split empty, YOLO would fail later with an error
    # that says nothing about the cause, so stop here with a clear message.
    counts = _count_split_images(augmented_path)
    if counts["train"] == 0 or counts["val"] == 0:
        show_error_window(
            "The prepared dataset is empty, so training cannot start."
            + "\n\n"
            + f"Usable training images: {counts['train']}\n"
            + f"Usable validation images: {counts['val']}\n\n"
            + "Every image needs a YOLO label file matching the detected "
            + f"task ('{task}').",
            title="No Usable Training Data",
        )
        _clean_temp_dataset(dataset_path)
        return None

    yaml_file = create_yaml(tempfile.gettempdir())

    if not suppress_instructions:
        show_instructions(
            "Dataset preparation complete!\n\n"
            "The training process will now begin using the specified parameters.\n\n"
            "This may take some time depending on your hardware and dataset size.\n\n"
            "There will be two plots in the next screen. It may take some time for data to start appearing on them, depending on your dataset size, model size, and hardware capabilities.\n\n"
            "You can hover over points in the plots to see exact values."
        )

    try:
        results = run_training(
            yaml_file,
            model_save_dir,
            model_size,
            task=task,
            prev_model_path=previous_model_path,
        )
    finally:
        # Always clear the temporary dataset, even when training failed, so it
        # cannot contaminate the next run.
        _clean_temp_dataset(dataset_path)

    if results is None:
        # run_training has already reported the failure.
        return None

    if not suppress_instructions:
        show_instructions(
            "Training complete!\n\n"
            f"Trained model saved to: {model_save_dir}\n\n"
            "YOLOEZ will now close."
        )

    return results


def _clean_temp_dataset(dataset_path):
    """Remove the temporary dataset, its augmented copies and the generated YAML."""
    temp_root = os.path.split(dataset_path)[0]

    dataset_yaml_path = os.path.join(temp_root, "dataset.yaml")
    if os.path.exists(dataset_yaml_path):
        os.remove(dataset_yaml_path)

    for name in ("augmented_dataset", "yolo_dataset"):
        path = os.path.join(temp_root, name)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


def _count_split_images(dataset_root):
    """Count the images present in each split of a prepared dataset."""
    counts = {}
    for split in ("train", "val"):
        split_dir = os.path.join(dataset_root, "images", split)
        counts[split] = len(os.listdir(split_dir)) if os.path.isdir(split_dir) else 0
    return counts
