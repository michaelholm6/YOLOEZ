import pytest
from main import main as run_main 

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
        "output_folder": tmp_path,
        "save_unlabeled_images": False
    }


    monkeypatch.setattr("labelling_workflow.main_labelling.show_instructions", lambda **kwargs: None)
    monkeypatch.setattr("labelling_workflow.main_labelling.annotate_images", lambda paths: ({}, 1))
    monkeypatch.setattr("labelling_workflow.main_labelling.crop_and_mask_images", lambda images, areas: images)
    monkeypatch.setattr("labelling_workflow.main_labelling.run_contour_editor", lambda *args, **kwargs: [{}]*len(test_inputs["image_paths"]))
    monkeypatch.setattr("labelling_workflow.main_labelling.run_box_editor", lambda *args, **kwargs: [{}]*len(test_inputs["image_paths"]))
    monkeypatch.setattr("labelling_workflow.main_labelling.run_yolo_on_crops", lambda *args, **kwargs: [{}]*len(test_inputs["image_paths"]))
    monkeypatch.setattr("labelling_workflow.main_labelling.save_segmentation_results", lambda *args, **kwargs: None)
    monkeypatch.setattr("labelling_workflow.main_labelling.save_box_results", lambda *args, **kwargs: None)

    run_labeling_workflow(suppress_instructions=True, test_inputs=test_inputs)


    assert True
    
from training_workflow.main_training import run_training_workflow 

def test_training_workflow_runs_without_gui(monkeypatch, tmp_path):

    test_inputs = {
        "train_split": 0.8,
        "task": "segment", 
        "dataset_folder": "images", 
        "transformations": [],
        "number_of_augs": 1,
        "save_folder": tmp_path,
        "model_size": "nano",
        "prev_model_path": None,
    }


    monkeypatch.setattr("training_workflow.main_training.show_instructions", lambda *args, **kwargs: None)
    monkeypatch.setattr("training_workflow.main_training.split_dataset", lambda path, **kwargs: path)
    monkeypatch.setattr("training_workflow.main_training.augment_yolo_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr("training_workflow.main_training.create_yaml", lambda output_dir: str(tmp_path / "dataset.yaml"))
    monkeypatch.setattr("training_workflow.main_training.run_training", lambda *args, **kwargs: {"mock_result": True})


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


    monkeypatch.setattr("inference_workflow.main_inference.show_instructions", lambda *args, **kwargs: None)
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
    monkeypatch.setattr("inference_workflow.main_inference.run_yolo_inference", lambda *args, **kwargs: None)


    result = run_inference_workflow(suppress_instructions=True, test_inputs=test_inputs)


    assert True
    
from training_workflow.get_user_inputs_training import YOLOTrainingDialog


@pytest.mark.timeout(10)
def test_dialog_can_open_and_close(qtbot):
    """
    Smoke test: the dialog should be able to be constructed,
    shown, and closed without crashing.
    """

    dlg = YOLOTrainingDialog()
    dlg.show()
    qtbot.wait(200)
    dlg.close()
    qtbot.wait(100)

    assert True
    

    


