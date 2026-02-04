# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

from ultralytics import YOLO
import numpy as np
import cv2
from utils import show_error_window


def extract_clean_contours_from_mask(mask):
    """
    mask: uint8 binary image (0 or 255)
    returns: list of well-ordered OpenCV contours
    """

    # 1. Morphological cleanup (critical for cracks)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2. Find contours (ordered by traversal)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    clean_contours = []
    for cnt in contours:
        if len(cnt) < 5:
            continue  # too small / degenerate
        clean_contours.append(cnt)

    return clean_contours


def run_yolo_on_crops(
    images_dict, model_path, confidence_threshold=0.5, annotation_mode="bounding_box"
):

    model = YOLO(model_path)

    model_type = getattr(model.model, "model_type", None)
    if model_type is None:
        if hasattr(model.model, "masks") or "Seg" in type(model.model).__name__:
            model_type = "segmentation"
        else:
            model_type = "bounding_box"

    if model_type == "segmentation" and annotation_mode != "segmentation":
        show_error_window(
            f"Bootstrapping YOLO model is segmentation but annotation_mode='{annotation_mode}'",
            title="Model Type Mismatch",
        )
    elif model_type == "bounding_box" and annotation_mode != "bounding_box":
        show_error_window(
            f"Bootstrapping YOLO model is bounding_box but annotation_mode='{annotation_mode}'",
            title="Model Type Mismatch",
        )

    results_dict = {}

    for img_path, img in images_dict.items():
        results = model.predict(img, verbose=False, conf=confidence_threshold)
        result = results[0]
        contours_list = []

        if annotation_mode == "segmentation":
            if result.masks is None or result.boxes is None:
                results_dict[img_path] = []
                continue

            masks = result.masks.data.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            orig_h, orig_w = result.orig_shape

            for mask, conf in zip(masks, confs):
                if conf < confidence_threshold:
                    continue

                binary = (mask > 0.5).astype(np.uint8) * 255
                binary = cv2.resize(
                    binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

                contours = extract_clean_contours_from_mask(binary)
                contours_list.extend(contours)

        else:  # bounding box mode
            if result.boxes is not None:
                for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                    if conf < confidence_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    contour = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
                    ).reshape((-1, 1, 2))
                    contours_list.append(contour)

        results_dict[img_path] = contours_list

    return results_dict
