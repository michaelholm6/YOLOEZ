import cv2
import numpy as np

def crop_and_mask_images(image_paths, aois_dict):
    """
    Mask images so that only the AOI polygon remains visible.
    Everything outside the AOI is black.
    Image dimensions are preserved.

    Args:
        image_paths (list[str]): List of image file paths.
        aois_dict (dict): {image_path: list of [x, y] polygon points}.

    Returns:
        dict: {image_path: masked_image (numpy array)}.
    """
    masked_results = {}

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}: failed to load.")
            continue

        aoi = aois_dict.get(img_path, [])
        if not aoi or len(aoi) < 3:
            # No valid AOI: return original image
            masked_results[img_path] = img.copy()
            continue

        aoi_np = np.array(aoi, dtype=np.int32)

        # 1. Create full-size mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [aoi_np], 255)

        # 2. Apply mask (outside AOI becomes black)
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        masked_results[img_path] = masked_img

    return masked_results
