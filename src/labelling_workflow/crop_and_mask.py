# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import cv2
import numpy as np

def crop_and_mask_images(image_paths, aois_dict):
    """
    Crop images to the smallest bounding rectangle around AOIs,
    masking everything outside the AOI polygon to black.
    If no AOI exists for an image, return the original image.

    Args:
        image_paths (list[str]): List of image file paths.
        aois_dict (dict): {image_path: list of [x, y] polygon points}.

    Returns:
        dict: {image_path: cropped_image (numpy array)}.
    """
    cropped_results = {}

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}: failed to load.")
            continue

        aoi = aois_dict.get(img_path, [])
        if not aoi or len(aoi) < 3:
            # No valid AOI: return the original image
            cropped_results[img_path] = img.copy()
            continue

        aoi_np = np.array(aoi, dtype=np.int32)

        # 1. Compute smallest bounding box around AOI
        x, y, w, h = cv2.boundingRect(aoi_np)

        # 2. Crop image
        cropped = img[y:y+h, x:x+w].copy()

        # 3. Create mask in cropped coordinates
        shifted_aoi = aoi_np - [x, y]
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [shifted_aoi], 255)

        # 4. Apply mask
        masked_cropped = cv2.bitwise_and(cropped, cropped, mask=mask)

        cropped_results[img_path] = masked_cropped

    return cropped_results
