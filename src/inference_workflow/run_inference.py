# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import cv2
import json
import numpy as np
import os
import torch
import gc
from PyQt5 import QtWidgets, QtCore

def run_yolo_inference(
    model,
    cropped_images,
    save_path,
    conf,
    parent=None
):
    num_images = len(cropped_images)
    progress = QtWidgets.QProgressDialog(
        "Running inference...",
        "Cancel",
        0,
        num_images,
        parent
    )
    progress.setWindowTitle("Inference")
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    progress.setMinimumSize(400, 120)

    os.makedirs(save_path, exist_ok=True)

    for i, img in enumerate(cropped_images.values(), 1):
        QtWidgets.QApplication.processEvents()
        if progress.wasCanceled():
            break

        result = model(np.array(img), conf=conf)

        annotated_img = result[0].plot(img=img, conf=True)

        img_path = os.path.join(save_path, f"result_{i}.jpg")
        tmp_path = os.path.join(save_path, f".result_{i}.jpg")

        annotated_img = np.ascontiguousarray(annotated_img, dtype=np.uint8)

        if not cv2.imwrite(tmp_path, annotated_img):
            raise RuntimeError(f"Failed to write temp file {tmp_path}")

        os.replace(tmp_path, img_path)
        del annotated_img

        det_list = []
        boxes = result[0].boxes
        names = result[0].names
        masks = getattr(result[0], "masks", None)

        for j, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            label = names[cls_id] if names and cls_id in names else str(cls_id)

            polygon = None
            if masks:
                mask_data = masks.data[j].cpu().numpy()
                contours, _ = cv2.findContours(
                    mask_data.astype('uint8'),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    polygon = contours[0].reshape(-1, 2).tolist()
                del mask_data

            det_list.append({
                "image_index": i,
                "class_id": cls_id,
                "class_name": label,
                "confidence": conf_val,
                "bbox": xyxy,
                "segmentation": polygon
            })

        json_path = os.path.join(save_path, f"result_{i}.json")
        with open(json_path, "w") as f:
            json.dump(det_list, f, indent=4)

        del result, boxes, masks, det_list
        gc.collect()
        progress.setValue(i)

    progress.close()