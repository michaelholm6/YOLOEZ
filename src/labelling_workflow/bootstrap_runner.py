# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

from ultralytics import YOLO
import numpy as np
from utils import show_error_window

def run_yolo_on_crops(images_dict, model_path, confidence_threshold=0.5, annotation_mode='bounding_box'):
    
    model = YOLO(model_path)

    # Check model type
    model_type = getattr(model.model, 'model_type', None)
    if model_type is None:
        # fallback: segmentation models often have no 'head'
        if hasattr(model.model, 'masks') or 'Seg' in type(model.model).__name__:
            model_type = 'segmentation'
        else:
            model_type = 'bounding_box'

    if model_type == 'segmentation' and annotation_mode != 'segmentation':
        show_error_window(
            f"Bootstrapping YOLO model is segmentation but annotation_mode='{annotation_mode}'",
            title="Model Type Mismatch"
        )
    elif model_type == 'bounding_box' and annotation_mode != 'bounding_box':
        show_error_window(
            f"Bootstrapping YOLO model is bounding_box but annotation_mode='{annotation_mode}'",
            title="Model Type Mismatch"
        )

    results_dict = {}

    for img_path, img in images_dict.items():
        results = model.predict(img, verbose=False, conf=confidence_threshold, save=False)
        result = results[0]  # one per image
        contours_list = []

        if annotation_mode == 'segmentation':
            if result.masks is not None and hasattr(result.masks, 'xy'):
                for i, poly in enumerate(result.masks.xy):
                    # check confidence if available
                    if hasattr(result.masks, 'conf') and result.masks.conf is not None:
                        if result.masks.conf[i] < confidence_threshold:
                            continue
                    if len(poly) == 0:
                        continue
                    contour = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    contours_list.append(contour)
        else:  # detection
            if result.boxes is not None and len(result.boxes) > 0:
                for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                    if conf < confidence_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    contour = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.int32).reshape((-1, 1, 2))
                    contours_list.append(contour)

        results_dict[img_path] = contours_list

    return results_dict
