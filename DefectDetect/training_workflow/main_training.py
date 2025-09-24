from DefectDetect.training_workflow.dataset_chooser import choose_dataset
from DefectDetect.training_workflow.train_test_split import get_train_test_split
from DefectDetect.training_workflow.augmentation_options import get_augmentations
from DefectDetect.training_workflow.save_model import get_save_path, save_trained_model
from ultralytics import YOLO
import os
from DefectDetect.utils import show_instructions
from DefectDetect.training_workflow.apply_augmentations import apply_augmentations
from DefectDetect.training_workflow.create_yaml import create_yaml
from DefectDetect.training_workflow.train_model import run_training
import shutil

def run_training_workflow(suppress_instructions=False):
    if not suppress_instructions:
        show_instructions(
            "Welcome to the Training Workflow!\n\n"
            "You will first need to decide what train / test split to use for your dataset.\n\n"
            "This determines how much data is used for training the model vs. testing its performance.\n\n"
            "A common choice is 80% for training and 20% for testing.\n\n"
            "Press escape to close the program at any time."
        )

    split = get_train_test_split()

    if not suppress_instructions:
        show_instructions(
            "Next, you will choose the dataset to use for training.\n\n"
            "Select the folder containing your images and corresponding label files.\n\n"
            "These can be created using the Labeling Workflow.\n\n"
            "The dataset will be prepared for YOLO training, assuming all data belongs to a single class called 'defect'.\n\n"
            "Press escape to close the program at any time."
        )

    dataset_path = choose_dataset(train_split=split)

    if not suppress_instructions:
        show_instructions(
            "Finally, you can choose whether to apply data augmentations during training.\n\n"
            "Data augmentations can help improve model robustness by introducing variations in the training data.\n\n"
            "Common augmentations include rotations, flips, scaling, color adjustments, noise, and blur.\n\n"
            "You can select which augmentations to apply in the next dialog.\n\n"
            "Press escape to close the program at any time."
        )

    augs, number_of_augs = get_augmentations()
    
    apply_augmentations(augs, dataset_path, os.path.join(os.path.split(dataset_path)[0], "augmented_dataset"), number_of_augs)
    
    yaml_file = create_yaml(os.path.split(dataset_path)[0])
    
    results = run_training("yolov8n-seg.pt", yaml_file)
    
    os.remove(os.path.join(os.path.split(dataset_path)[0], "dataset.yaml"))
    shutil.rmtree(os.path.join(os.path.split(dataset_path)[0], "augmented_dataset"))
    shutil.rmtree(os.path.join(os.path.split(dataset_path)[0], "yolo_dataset"))

    if not suppress_instructions:
        show_instructions(
            "Training complete!\n\n"
            "You can now choose where to save the trained model.\n\n"
            "It's recommended to save it in a dedicated folder for easy access later."
        )

    # --- Save model ---
    save_path = get_save_path()
    
    if save_path:
        best_model = results.save_dir / "weights" / "best.pt"
        save_trained_model(best_model, save_path)

    return results
