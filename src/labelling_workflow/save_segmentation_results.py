# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import os
import cv2
import numpy as np

def save_segmentation_results(
    image_paths,
    contours_dict,
    line_thickness,
    output_dir,
    areas_of_interest,
    save_yolo_dataset=True,
    save_unlabeled_images=False
):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    contour_vis_dir = os.path.join(output_dir, "contour_visualizations")
    os.makedirs(contour_vis_dir, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}: failed to load.")
            continue

        cnts = contours_dict.get(img_path, [])
        cnts = [np.array(c, dtype=np.int32) for c in cnts]

        # 🔹 Only skip if unlabeled images are NOT allowed
        if len(cnts) == 0 and not save_unlabeled_images:
            print(f"{img_path}: no contours → skipping sample.")
            continue

        # ---------- AOI handling ----------
        area_of_interest = areas_of_interest.get(img_path, [])
        has_aoi = len(area_of_interest) > 0

        if has_aoi:
            aoi_np = np.array(area_of_interest, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(aoi_np)

            cropped = img[y:y+h, x:x+w].copy()
            mask = np.zeros(cropped.shape[:2], dtype=np.uint8)

            shifted_aoi = aoi_np - [x, y]
            cv2.fillPoly(mask, [shifted_aoi], 255)

            base_img = cv2.bitwise_and(cropped, cropped, mask=mask)
        else:
            x, y, w, h = 0, 0, img.shape[1], img.shape[0]
            base_img = img.copy()

        # ---------- Draw contours ----------
        contour_vis = base_img.copy()
        for cnt in cnts:
            cv2.drawContours(
                contour_vis,
                [cnt],
                -1,
                (0, 255, 0),
                line_thickness
            )

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # ---------- Save contour visualization ----------
        contour_img_path = os.path.join(
            contour_vis_dir,
            f"{img_name}_contours.png"
        )
        cv2.imwrite(contour_img_path, contour_vis)

        # ---------- YOLO dataset ----------
        if save_yolo_dataset:
            yolo_dir = os.path.join(output_dir, f"yolo_data_{img_name}")
            os.makedirs(yolo_dir, exist_ok=True)

            # Save image
            img_out_path = os.path.join(yolo_dir, f"{img_name}.png")
            cv2.imwrite(img_out_path, base_img)

            # Save label file (may be empty)
            txt_path = os.path.join(yolo_dir, f"{img_name}.txt")
            h_img, w_img = base_img.shape[:2]

            with open(txt_path, "w") as f:
                for cnt in cnts:
                    polygon = []
                    for pt in cnt:
                        px, py = pt[0]
                        polygon.extend([px / w_img, py / h_img])

                    if len(polygon) < 6:
                        continue

                    class_id = 0
                    poly_str = " ".join(f"{p:.6f}" for p in polygon)
                    f.write(f"{class_id} {poly_str}\n")

        results.append(contour_vis)

    return results
