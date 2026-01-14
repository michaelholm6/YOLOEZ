from ultralytics import YOLO
import gc
import os
from utils import show_instructions
from YOLOEZ.labelling_workflow.area_of_interest_marking import annotate_images
from YOLOEZ.inference_workflow.save_results import get_save_path
from YOLOEZ.labelling_workflow.crop_and_mask import crop_and_mask_images
from YOLOEZ.inference_workflow.get_user_inputs_inference import get_user_inference_inputs
import numpy as np
import cv2
import json
from PyQt5 import QtWidgets, QtCore
from YOLOEZ.inference_workflow.run_inference import run_yolo_inference

def run_inference_workflow(trained_model_path=None, suppress_instructions=False):
    """
    Main method for performing inference with a trained YOLO model.
    
    Args:
        trained_model_path (str): Path to the trained YOLO .pt/.pth model file.
        suppress_instructions (bool): If True, skips showing instructions to the user.
    """
    
    if not suppress_instructions:
        show_instructions(
            "Welcome to the Inference Workflow!\n\n"
            "In this workflow, you will select images, optionally crop them to areas of interest, "
            "and run inference using a trained YOLO model to detect objects in the images.\n\n"
            "In the following screen, you will provide the necessary inputs to proceed."
        )
    
    inputs = get_user_inference_inputs()
    image_paths = inputs["image_paths"]
    
    if not suppress_instructions:
        show_instructions(
            f"You selected {len(image_paths)} image(s).\n\n"
            "Next, you can optionally crop each image to an area of interest.\n"
            "If you do not wish to crop an image, simply leave it unmarked and proceed.\n\n"
            
            "When finished, close the window to proceed."
        )
    
    areas_of_interest, _ = annotate_images(image_paths)
    
    cropped_images = crop_and_mask_images(image_paths, areas_of_interest)
    
    if not suppress_instructions:
        show_instructions(
            f"Preprocessing complete! {len(cropped_images)} image(s) ready for inference.\n\n"
        )
    
    # --- Step 3: Load trained model ---
    trained_model_path = inputs["YOLO_model"]
    
    model = YOLO(trained_model_path)
        
    save_path = inputs["output_folder"]
    
    os.makedirs(save_path, exist_ok=True)
    
    run_yolo_inference(
        model, cropped_images, save_path, inputs["YOLO_confidence"]
    )
