import cv2
import numpy as np

def crop_and_mask_images(image_paths, aois_dict):
    """
    Mask images so that only the AOI polygons remain visible.
    Everything outside AOIs is black.
    Image dimensions are preserved.

    Args:
        image_paths (list[str]): List of image file paths.
        aois_dict (dict): {image_path: list of polygons}, 
                          each polygon is a list of [x, y] points.

    Returns:
        dict: {image_path: masked_image (numpy array)}.
    """
    masked_results = {}

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}: failed to load.")
            continue

        polygons = aois_dict.get(img_path, [])
        if not polygons:
            # No AOIs: return original image
            masked_results[img_path] = img.copy()
            continue

        # 1. Create full-size mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 2. Draw each polygon
        for poly in polygons:
            if len(poly) >= 3:
                poly_np = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [poly_np], 255)

        # 3. Apply mask (outside AOIs becomes black)
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        masked_results[img_path] = masked_img

    return masked_results
