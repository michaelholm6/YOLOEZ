# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

from labelling_workflow.image_gatherer import *
from labelling_workflow.area_of_interest_marking import *
from labelling_workflow.save_segmentation_results import *
from labelling_workflow.edit_contour_points import *
from labelling_workflow.get_user_inputs_labelling import *
from labelling_workflow.edit_box_points import *
from labelling_workflow.save_box_results import *
from labelling_workflow.crop_and_mask import *
from labelling_workflow.bootstrap_runner import *
from utils import show_instructions

def run_labeling_workflow(suppress_instructions=False):
    
    if not suppress_instructions:
        instructions = (
        "Welcome to the Labelling Workflow!\n\n"
        "- You will be guided through a series of steps to label your images.\n"
        "- Please follow the on-screen instructions at each step.\n"
        "- You can press 'Esc' at any time to exit the workflow.\n\n"
        "Click 'OK' to begin."
    )
        show_instructions(message=instructions)
    
    args = get_user_labelling_inputs()
        
    images = args["image_paths"]
    
    if not suppress_instructions:
        instructions = (
        "Instructions for Marking the Area of Interest:\n\n"
        "- You will now select areas of interest for each image. You will do this by drawing a polygon around the area of interest.\n"
        "- The polygon should outline the area where you want to detect objects.\n"
        "- Left click to add points and outline your polygon.\n"
        "- Press 'C' to close the area of interest polygon.\n"
        "- Press 'R' to reset points.\n"
        "- If you want the entire image as the area of interest, just don't draw a polygon on that image.\n"
        "- Press 'Esc' to cancel and close the application.\n"
        "- When finished, close the window.\n"
    )
        show_instructions(message=instructions)
        
    areas_of_interest, line_thickness = annotate_images(args["image_paths"])
    
    cropped_images = crop_and_mask_images(images, areas_of_interest)
    
    if args['bootstrapping_model'] != None:
        
        detection_results = run_yolo_on_crops(cropped_images, args['bootstrapping_model'], args['bootstrapping_confidence'], args['annotation_mode'])
    
    
    if args['annotation_mode'] == 'segmentation':
        
        if not suppress_instructions:
            instructions = (
                "Instructions for Editing Contours:\n\n"
                "- You will now manually edit or create contours on the image.\n"
                "- Left click and hold to select points by circling them with a lasso.\n"
                "- Right click and drag to pan the image.\n"
                "- Press 'C' to create a new contour by clicking to add points. Close the contour by pressing 'C' again.\n"
                "- Press 'D' to delete selected points.\n"
                "- Press 'U' to undo the last action. NOTE: You cannot undo creating individual points while in create mode, but you can move/delete them after exiting create mode.\n"
                "- Press 'S' to scale selected points (drag mouse relative to center).\n"
                "- Press 'M' to move selected points (drag mouse).\n"
                "- Press 'R' to rotate selected points (drag mouse relative to center).\n"
                "- Use the mouse wheel to zoom in and out.\n"
                "- Press 'Esc' to cancel and close the application.\n"
                "- Press the same key again or left click to exit a mode.\n\n"
                "When you're done, simply close the window to continue."
            )
            show_instructions(message=instructions)
    
        contours = run_contour_editor(cropped_images, line_thickness, detected_contours=detection_results if args['bootstrapping_model'] != None else None)
        
    else:
        
        if not suppress_instructions:
            instructions = (
                "Instructions for Editing Bounding Boxes:\n\n"
                "- You will now manually edit or create bounding boxes on the image.\n"
                "- Left click and hold to select boxes by circling them with a lasso.\n"
                "- Right click and drag to pan the image.\n"
                "- Press 'C' to create a new bounding box by clicking and dragging.\n"
                "- Press 'D' to delete selected points (this will delete all boxes associated with those points).\n"
                "- Press 'U' to undo the last action.\n"
                "- Use the mouse wheel to zoom in and out.\n"
                "- Press 'Esc' to cancel and close the application.\n"
                "- Press the same key again or left click to exit a mode.\n\n"
                "When you're done, simply close the window to continue."
                
            )
            show_instructions(message=instructions)
        
        contours = run_box_editor(cropped_images, line_thickness, initial_boxes=detection_results if args['bootstrapping_model'] != None else None)
        
    save_yolo = args['YOLO_true']
        
    if args['annotation_mode'] == 'segmentation':
    
        save_segmentation_results(images, contours, line_thickness, args["output_folder"], areas_of_interest, save_yolo, args['save_unlabeled_images'])
        
    else:
    
        save_box_results(images, contours, line_thickness, args["output_folder"], areas_of_interest, save_yolo, args['save_unlabeled_images'])  
        
    if not suppress_instructions:
        message = "Labeling workflow complete! Results have been saved to the specified output folder."
        show_instructions(message=message, title="Workflow Complete")  