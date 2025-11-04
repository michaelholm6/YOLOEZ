from ultralytics import YOLO
import os
from DefectDetect.utils import show_instructions
from DefectDetect.inference_workflow.image_chooser import choose_image_folder
from DefectDetect.inference_workflow.preprocess_images import preprocess_images
from DefectDetect.inference_workflow.get_model_path import get_model_path
from DefectDetect.inference_workflow.save_results import get_save_path, postprocess_and_save_results

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
            "First, you will select a folder containing images for inference.\n\n"
            "Supported formats include PNG, JPG, BMP, and TIFF."
        )
    
    image_paths = choose_image_folder()
    if not image_paths:
        print("No images selected. Exiting.")
        return
    
    if not suppress_instructions:
        show_instructions(
            f"You selected {len(image_paths)} image(s).\n\n"
            "Next, you can optionally crop each image to an area of interest.\n"
            "If you do not wish to crop, simply close the window."
        )
    
    # --- Step 2: Optional preprocessing ---
    original_cropped, preprocessed_images = preprocess_images(image_paths)
    
    if not suppress_instructions:
        show_instructions(
            f"Preprocessing complete! {len(preprocessed_images)} image(s) ready for inference.\n\n"
            "Next, you will select the trained YOLO model to use for inference."
        )
    
    # --- Step 3: Load trained model ---
    if not trained_model_path:
        trained_model_path = get_model_path()
    
    model = YOLO(trained_model_path)
    
    if not suppress_instructions:
        show_instructions(
            f"Running inference on {len(preprocessed_images)} image(s) using model:\n{trained_model_path}\n\n"
            "This may take a few moments depending on image size and model complexity."
        )
        
    results = model(preprocessed_images)
    
    if not suppress_instructions:
        show_instructions(
            "Inference complete!\n\n"
            "You can now choose where to save the results for later analysis."
        )
    
    # --- Step 5: Save results ---
    save_path = get_save_path()
    if save_path:
        postprocess_and_save_results(results, original_cropped, save_path)