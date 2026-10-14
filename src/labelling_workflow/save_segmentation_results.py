# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import os
import cv2
import numpy as np
from utils import write_image


def _write_image(path, image):
    """Write an image, reporting failures instead of letting them pass silently."""
    if not write_image(path, image):
        print(f"Warning: failed to write {path}")
        return False
    return True


def save_segmentation_results(
    image_paths,
    contours_dict,
    masked_images_dict,
    line_thickness,
    output_dir,
    save_yolo_dataset=True,
    save_unlabeled_images=False,
):
    """
    Saves segmentation results for AOI-masked images (black outside AOI).

    Args:
        image_paths (list[str])
        contours_dict (dict[str, list[np.ndarray]])
        masked_images_dict (dict[str, np.ndarray])  # full-size, black outside AOI
        line_thickness (int)
        output_dir (str)
    """
    os.makedirs(output_dir, exist_ok=True)

    contour_vis_dir = os.path.join(output_dir, "contour_visualizations")
    mask_dir = os.path.join(output_dir, "segmentation_masks")
    os.makedirs(contour_vis_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    results = []

    for img_path in image_paths:
        base_img = masked_images_dict.get(img_path, None)
        if base_img is None:
            print(f"Skipping {img_path}: masked image not found.")
            continue

        # "or []" rather than a .get default: an editor may store None for an
        # image the user never opened, and the key exists in that case.
        cnts = contours_dict.get(img_path) or []
        cnts = [np.array(c, dtype=np.int32) for c in cnts]

        # Drop contours that cannot form a polygon.  This happens before the
        # save_unlabeled_images decision: they produce no label line, and
        # writing an empty .txt would add a negative sample the user asked not
        # to include.
        cnts = [c for c in cnts if len(c) >= 3]

        if len(cnts) == 0 and not save_unlabeled_images:
            print(f"{img_path}: no contours, skipping.")
            continue

        h_img, w_img = base_img.shape[:2]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # --------------------------------------------------
        # 1. Contour visualization (for debugging)
        # --------------------------------------------------
        contour_vis = base_img.copy()
        for cnt in cnts:
            cv2.drawContours(contour_vis, [cnt], -1, (0, 255, 0), line_thickness)

        contour_img_path = os.path.join(contour_vis_dir, f"{img_name}_contours.png")
        _write_image(contour_img_path, contour_vis)

        # --------------------------------------------------
        # 2. Binary segmentation mask
        #    White = object, Black = background
        # --------------------------------------------------
        seg_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        for cnt in cnts:
            cv2.fillPoly(seg_mask, [cnt], 255)

        mask_path = os.path.join(mask_dir, f"{img_name}_mask.png")
        _write_image(mask_path, seg_mask)

        # --------------------------------------------------
        # 3. YOLO polygon dataset (full image coordinates)
        # --------------------------------------------------
        if save_yolo_dataset:
            yolo_dir = os.path.join(output_dir, "yolo_dataset")
            os.makedirs(yolo_dir, exist_ok=True)

            img_out_path = os.path.join(yolo_dir, f"{img_name}.png")
            _write_image(img_out_path, base_img)

            txt_path = os.path.join(yolo_dir, f"{img_name}.txt")
            with open(txt_path, "w") as f:
                for cnt in cnts:
                    polygon = []
                    for pt in cnt:
                        px, py = pt[0]
                        polygon.extend([px / w_img, py / h_img])

                    class_id = 0
                    poly_str = " ".join(f"{p:.6f}" for p in polygon)
                    f.write(f"{class_id} {poly_str}\n")

        results.append(contour_vis)

    return results
