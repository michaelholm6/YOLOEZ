from ultralytics import YOLO

def run_yolo_on_crops(images_dict, model_path):
    """
    Parameters
    ----------
    images_dict : dict
        Keys = original image paths (str)
        Values = cropped OpenCV images (numpy arrays)
    model_path : str
        Path to a trained Ultralytics YOLO model (.pt)

    Returns
    -------
    dict : { original_path: [cropped_image, detections] }
        detections will contain:
            - result.boxes for detection models
            - result.masks for segmentation models (None if not a seg model)
    """

    model = YOLO(model_path)

    results_dict = {}

    for img_path, cropped_img in images_dict.items():
        # Run inference
        results = model.predict(cropped_img, verbose=False)

        # YOLO returns a list of results (one per image)
        result = results[0]

        # Collect boxes / masks
        detections = {
            "boxes": result.boxes,  # always exists
            "masks": result.masks   # None for non-segmentation models
        }

        results_dict[img_path] = [cropped_img, detections]

    return results_dict