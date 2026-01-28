# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import os
import cv2
import numpy as np


def save_box_results(
    image_paths,
    boxes_dict,
    masked_images_dict,
    line_thickness,
    output_dir,
    save_yolo_dataset=True,
    save_unlabeled_images=False,
):
    """
    Save bounding-box results for AOI-masked images (black outside AOI).

    Args:
        image_paths (list[str])
        boxes_dict (dict[str, list[np.ndarray]])
        masked_images_dict (dict[str, np.ndarray])  # full-size, AOI-masked images
        line_thickness (int)
        output_dir (str)
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    box_vis_dir = os.path.join(output_dir, "box_visualizations")
    os.makedirs(box_vis_dir, exist_ok=True)

    for img_path in image_paths:
        base_img = masked_images_dict.get(img_path, None)
        if base_img is None:
            print(f"Skipping {img_path}: masked image not found.")
            continue

        boxes = boxes_dict.get(img_path, [])
        boxes = [np.array(b, dtype=np.int32) for b in boxes]

        if len(boxes) == 0 and not save_unlabeled_images:
            print(f"{img_path}: no boxes → skipping.")
            continue

        h_img, w_img = base_img.shape[:2]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # --------------------------------------------------
        # 1. Box visualization
        # --------------------------------------------------
        box_vis = base_img.copy()

        for box in boxes:
            pts = box.reshape(-1, 2)

            bx1 = int(np.clip(np.min(pts[:, 0]), 0, w_img - 1))
            by1 = int(np.clip(np.min(pts[:, 1]), 0, h_img - 1))
            bx2 = int(np.clip(np.max(pts[:, 0]), 0, w_img - 1))
            by2 = int(np.clip(np.max(pts[:, 1]), 0, h_img - 1))

            if bx2 <= bx1 or by2 <= by1:
                continue

            cv2.rectangle(
                box_vis, (bx1, by1), (bx2, by2), (0, 255, 0), int(line_thickness)
            )

        box_img_path = os.path.join(box_vis_dir, f"{img_name}_boxes.png")
        cv2.imwrite(box_img_path, box_vis)

        # --------------------------------------------------
        # 2. YOLO bounding-box dataset (full image coords)
        # --------------------------------------------------
        if save_yolo_dataset:
            yolo_dir = os.path.join(output_dir, "yolo_dataset")
            os.makedirs(yolo_dir, exist_ok=True)

            img_out_path = os.path.join(yolo_dir, f"{img_name}.png")
            cv2.imwrite(img_out_path, base_img)

            txt_path = os.path.join(yolo_dir, f"{img_name}.txt")
            with open(txt_path, "w") as f:
                for box in boxes:
                    pts = box.reshape(-1, 2)

                    bx1 = np.min(pts[:, 0])
                    by1 = np.min(pts[:, 1])
                    bx2 = np.max(pts[:, 0])
                    by2 = np.max(pts[:, 1])

                    bx1 = np.clip(bx1, 0, w_img - 1)
                    bx2 = np.clip(bx2, 0, w_img - 1)
                    by1 = np.clip(by1, 0, h_img - 1)
                    by2 = np.clip(by2, 0, h_img - 1)

                    bw = bx2 - bx1
                    bh = by2 - by1
                    if bw <= 0 or bh <= 0:
                        continue

                    cx = (bx1 + bx2) / 2 / w_img
                    cy = (by1 + by2) / 2 / h_img

                    class_id = 0
                    f.write(
                        f"{class_id} {cx:.6f} {cy:.6f} "
                        f"{bw / w_img:.6f} {bh / h_img:.6f}\n"
                    )

        results.append(box_vis)

    return results
