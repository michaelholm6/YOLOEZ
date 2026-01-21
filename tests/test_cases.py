import pytest
from main import main as run_main 

def test_main_runs_without_error(monkeypatch):
    monkeypatch.setattr("sys.exit", lambda *args, **kwargs: None)

    run_main(test_mode=True)

    assert True
    
from labelling_workflow.main_labelling import run_labeling_workflow

def test_labeling_workflow_runs_without_gui(tmp_path, monkeypatch):
    # Mock user inputs for the workflow
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
        "output_folder": tmp_path,
        "save_unlabeled_images": False
    }

    # Patch GUI functions so they don't actually pop up
    monkeypatch.setattr("labelling_workflow.main_labelling.show_instructions", lambda **kwargs: None)
    monkeypatch.setattr("labelling_workflow.main_labelling.annotate_images", lambda paths: ({}, 1))
    monkeypatch.setattr("labelling_workflow.main_labelling.crop_and_mask_images", lambda images, areas: images)
    monkeypatch.setattr("labelling_workflow.main_labelling.run_contour_editor", lambda *args, **kwargs: [{}]*len(test_inputs["image_paths"]))
    monkeypatch.setattr("labelling_workflow.main_labelling.run_box_editor", lambda *args, **kwargs: [{}]*len(test_inputs["image_paths"]))
    monkeypatch.setattr("labelling_workflow.main_labelling.run_yolo_on_crops", lambda *args, **kwargs: [{}]*len(test_inputs["image_paths"]))
    monkeypatch.setattr("labelling_workflow.main_labelling.save_segmentation_results", lambda *args, **kwargs: None)
    monkeypatch.setattr("labelling_workflow.main_labelling.save_box_results", lambda *args, **kwargs: None)

    # Run workflow in test mode
    run_labeling_workflow(suppress_instructions=True, test_inputs=test_inputs)

    # If we reach here, nothing crashed
    assert True
