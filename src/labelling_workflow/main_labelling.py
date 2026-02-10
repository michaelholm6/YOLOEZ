# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

from labelling_workflow.area_of_interest_marking import annotate_images
from labelling_workflow.save_segmentation_results import save_segmentation_results
from labelling_workflow.edit_contour_points import run_contour_editor
from labelling_workflow.get_user_inputs_labelling import get_user_labelling_inputs
from labelling_workflow.edit_box_points import run_box_editor
from labelling_workflow.save_box_results import save_box_results
from labelling_workflow.crop_and_mask import crop_and_mask_images
from labelling_workflow.bootstrap_runner import run_yolo_on_crops
from utils import show_instructions


def run_labeling_workflow(
    suppress_instructions=False, test_inputs=None  # new parameter
):
    if not suppress_instructions:
        instructions = (
            "Welcome to the Labelling Workflow!\n\n"
            "- You will be guided through a series of steps to label your images.\n"
            "- Please follow the on-screen instructions at each step.\n"
            "- You can close a window at any time to exit the workflow.\n\n"
            "Click 'OK' to begin."
        )
        show_instructions(message=instructions)

    if test_inputs is not None:
        args = test_inputs
    else:
        args = get_user_labelling_inputs()

    images = args["image_paths"]

    if not suppress_instructions:
        instructions = (
            "Instructions for Marking the Area of Interest:\n\n"
            "- You will now select areas of interest for each image. You will do this by drawing a polygon around the areas of interest.\n"
            "- This is meant to make labelling easier by blacking out areas outside the areas of interest, so you don't have to label the entire image, if there are"
            "lots of instances of your object of interest in the image.\n"
            "- The polygons should outline the areas where you want to detect objects.\n"
            "- Left click to add points (or click, hold, and drag) and outline your polygon.\n"
            "- Close a polygon by clicking on the first point again.\n"
            "- Press 'Ctrl+Z' to undo the last action.\n"
            "- Press 'Ctrl+Y' to redo the last undone action.\n"
            "- If you want the entire image as the area of interest, just don't draw a polygon on that image.\n"
            "- When finished, click the 'Finish' button.\n"
        )
        show_instructions(message=instructions)

    areas_of_interest, line_thickness = annotate_images(args["image_paths"])

    cropped_images = crop_and_mask_images(images, areas_of_interest)

    if args["bootstrapping_model"] != None:

        detection_results = run_yolo_on_crops(
            cropped_images,
            args["bootstrapping_model"],
            args["bootstrapping_confidence"],
            args["annotation_mode"],
        )

    if args["annotation_mode"] == "segmentation":

        if not suppress_instructions:
            instructions = (
                "Instructions for Editing Contours:\n\n"
                "- You will now manually edit or create contours on the image.\n"
                "- Left click and drag to select points by circling them with a lasso.\n"
                "- Right click and drag to pan the image.\n"
                "- Press 'C' to create a new contour by clicking (or clicking, holding, and dragging) to add points. Close the contour by selecting the first point again.\n"
                "- Press 'D' to delete selected points.\n"
                "- Press 'Ctrl+Z' to undo the last action. NOTE: You cannot undo creating individual points while in create mode, but you can move/delete them after exiting create mode.\n"
                "- Press 'S' to scale selected points (drag mouse relative to center).\n"
                "- Press 'M' to move selected points (drag mouse).\n"
                "- Press 'R' to rotate selected points (drag mouse relative to center).\n"
                "- Press 'U' to union selected contours into a single contour (i.e. combine them into one contour).\n"
                "- Use the mouse wheel to zoom in and out.\n"
                "- Press the same key again or left click to exit a mode.\n\n"
                "When you're done, simply close the window to continue."
            )
            show_instructions(message=instructions)

        contours = run_contour_editor(
            cropped_images,
            line_thickness,
            detected_contours=(
                detection_results if args["bootstrapping_model"] != None else None
            ),
        )

    else:

        if not suppress_instructions:
            instructions = (
                "Instructions for Editing Bounding Boxes:\n\n"
                "- You will now manually edit or create bounding boxes on the image.\n"
                "- Left click and hold to select boxes by circling them with a lasso.\n"
                "- Right click and drag to pan the image.\n"
                "- Press 'C' to create a new bounding box by clicking and dragging.\n"
                "- Press 'D' to delete selected points (this will delete all boxes associated with those points).\n"
                "- Press 'ctrl+Z' to undo the last action.\n"
                "- Use the mouse wheel to zoom in and out.\n"
                "- Press the same key again, left click, or press 'esc' to exit a mode.\n\n"
                "When you're done, simply close the window to continue."
            )
            show_instructions(message=instructions)

        contours = run_box_editor(
            cropped_images,
            line_thickness,
            initial_boxes=(
                detection_results if args["bootstrapping_model"] != None else None
            ),
        )

    save_yolo = args["YOLO_true"]

    if args["annotation_mode"] == "segmentation":

        save_segmentation_results(
            images,
            contours,
            cropped_images,
            line_thickness,
            args["output_folder"],
            save_yolo,
            args["save_unlabeled_images"],
        )

    else:

        save_box_results(
            images,
            contours,
            cropped_images,
            line_thickness,
            args["output_folder"],
            save_yolo,
            areas_of_interest,
        )

    if not suppress_instructions:
        show_instructions(
            "Labelling complete!\n\n"
            f"Results saved to: {args['output_folder']}\n\n"
            "YOLOEZ will now close."
        )
