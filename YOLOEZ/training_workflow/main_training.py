from YOLOEZ.training_workflow.train_test_split import get_train_test_split
from YOLOEZ.training_workflow.augmentation_options import get_augmentations
from YOLOEZ.training_workflow.save_model import get_save_root, save_trained_model
from ultralytics import YOLO
import os
from utils import show_instructions
from YOLOEZ.training_workflow.apply_augmentations import augment_yolo_dataset
from YOLOEZ.training_workflow.create_yaml import create_yaml
from YOLOEZ.training_workflow.train_model import run_training
from YOLOEZ.training_workflow.get_task import get_task
from YOLOEZ.training_workflow.get_user_inputs_training import get_training_inputs
from YOLOEZ.training_workflow.dataset_chooser import split_dataset
import shutil
import tempfile
from pathlib import Path

def run_training_workflow(suppress_instructions=False):
    if not suppress_instructions:
        show_instructions(
            "Welcome to the Training Workflow!\n\n"
            "First, you will provide inputs to set up your training session.\n"
            "You will select your dataset, training parameters, and augmentation options.\n\n"
            "After that, the training process will begin automatically.\n\n"
            "Press esc at any time to exit the program."
        )
        
    inputs = get_training_inputs()

    split = inputs['train_split']
    task = inputs['task']
    dataset_path = inputs['dataset_folder']
    augs = inputs['transformations']
    number_of_augs = inputs['number_of_augs']
    model_save_dir = inputs['save_folder']
    model_size = inputs['model_size']
    previous_model_path = inputs.get('prev_model_path', None)
    
    dataset_path = split_dataset(dataset_path, train_split=split)
    
    
    augment_yolo_dataset(dataset_path, augs, os.path.join(tempfile.gettempdir(), "augmented_dataset"), number_of_augs, task=task)
    
    yaml_file = create_yaml(tempfile.gettempdir())
    
    if not suppress_instructions:
        show_instructions(
            "Dataset preparation complete!\n\n"
            "The training process will now begin using the specified parameters.\n\n"
            "This may take some time depending on your hardware and dataset size.\n\n"
            "There will be two plots in the next screen. It make take some time for data to start appearing on them, depending on your dataset size, model size, and hardware capabilities.\n\n"
            "You can click on a line in the plots to see exact values at each epoch."
            )
    
    if previous_model_path is not None:
        results = run_training(yaml_file, model_save_dir, model_size, task=task, prev_model_path=previous_model_path)
        
    else:
        results = run_training(yaml_file, model_save_dir, model_size, task=task)
    
    os.remove(os.path.join(os.path.split(dataset_path)[0], "dataset.yaml"))
    shutil.rmtree(os.path.join(os.path.split(dataset_path)[0], "augmented_dataset"))
    shutil.rmtree(os.path.join(os.path.split(dataset_path)[0], "yolo_dataset"))

    return results
