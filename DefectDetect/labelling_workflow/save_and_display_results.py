import os
import cv2
import numpy as np

def save_and_display_results(image_paths, contours_dict, line_thickness, output_dir, areas_of_interest):
    """
    Save result images with contours drawn, and prepare YOLO segmentation dataset.

    Args:
        image_paths (list of str): List of image file paths.
        contours_dict (dict): {image_path: list of contours (np.ndarray)}.
        line_thickness (int): Thickness for drawing contours.
        output_dir (str): Directory to save results.
        areas_of_interest (dict): {image_path: list of points defining AOI}.
    
    Returns:
        list of np.ndarray: Result images with contours drawn.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}: failed to load.")
            continue

        # Get contours for this image (empty list if missing)
        cnts = contours_dict.get(img_path, [])
        cnts = [np.array(c, dtype=np.int32) for c in cnts]  # ensure all contours are numpy arrays

        # Handle missing or empty AOI
        area_of_interest = areas_of_interest.get(img_path, [])
        has_aoi = len(area_of_interest) > 0

        if has_aoi:
            aoi_np = np.array(area_of_interest, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(aoi_np)
            cropped = img[y:y+h, x:x+w].copy()

            mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
            shifted_aoi = aoi_np - [x, y]
            cv2.fillPoly(mask, [shifted_aoi], 255)
            cropped_masked = cv2.bitwise_and(cropped, cropped, mask=mask)

            result = cropped_masked.copy()
        else:
            # No AOI → just use the original full image
            print(f"{img_path}: no AOI found, using full image.")
            x, y, w, h = 0, 0, img.shape[1], img.shape[0]
            cropped = img.copy()
            result = img.copy()

        # === Draw contours ===
        for cnt in cnts:
            shifted_cnt = cnt - [x, y]
            cv2.drawContours(result, [shifted_cnt], -1, (0, 255, 0), line_thickness)

        # === Save result image ===
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{img_name}_result.png")
        cv2.imwrite(output_path, result)

        # === Prepare YOLO dataset folder ===
        yolo_dir = os.path.join(output_dir, f"yolo_data_{img_name}")
        os.makedirs(yolo_dir, exist_ok=True)

        # Save image (masked if AOI exists, otherwise original)
        original_img_path = os.path.join(yolo_dir, f"{img_name}.png")
        cv2.imwrite(original_img_path, cropped if not has_aoi else cropped_masked)

        # === Save YOLO segmentation annotation ===
        txt_path = os.path.join(yolo_dir, f"{img_name}.txt")
        ch, cw = cropped.shape[:2]

        with open(txt_path, "w") as f:
            for cnt in cnts:
                shifted_cnt = cnt - [x, y]
                polygon = []
                for point in shifted_cnt:
                    px, py = point[0]
                    nx = px / cw
                    ny = py / ch
                    polygon.extend([nx, ny])

                if len(polygon) < 6:
                    continue  # skip invalid polygons

                class_id = 0
                polygon_str = " ".join([f"{p:.6f}" for p in polygon])
                f.write(f"{class_id} {polygon_str}\n")

        results.append(result)

    return results
