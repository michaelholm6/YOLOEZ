import os
import cv2
import numpy as np

def save_and_display_results(image, contours, line_thickness, output_path, area_of_interest):
    result = image.copy()

    # === Step 1: Crop image to bounding box of AOI ===
    aoi_np = np.array(area_of_interest, dtype=np.int32)  # shape (N,2)
    x, y, w, h = cv2.boundingRect(aoi_np)
    cropped = image[y:y+h, x:x+w].copy()

    # === Step 2: Create mask for AOI inside cropped region ===
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    shifted_aoi = aoi_np - [x, y]   # shift AOI coords into cropped image space
    cv2.fillPoly(mask, [shifted_aoi], 255)

    # Apply mask: keep AOI, set rest to black
    cropped_masked = cv2.bitwise_and(cropped, cropped, mask=mask)

    # === Step 3: Draw contours on result (shifted into cropped space) ===
    result = cropped_masked.copy()
    for cnt in contours:
        shifted_cnt = cnt - [x, y]  # shift contour coords into cropped image space
        color = (0, 255, 0)
        cv2.drawContours(result, [shifted_cnt], -1, color, line_thickness)

    # Save drawn result
    cv2.imwrite(output_path, result)

    # === Step 4: Prepare YOLO dataset folder ===
    base_dir = os.path.dirname(output_path)
    last_dir = os.path.splitext(os.path.basename(output_path))[0]
    yolo_dir = os.path.join(base_dir, f"yolo_data_{last_dir}")
    os.makedirs(yolo_dir, exist_ok=True)

    # Save cropped & masked image
    img_name = os.path.basename(output_path)
    original_img_path = os.path.join(yolo_dir, img_name)
    cv2.imwrite(original_img_path, cropped_masked)

    # === Step 5: Save YOLO segmentation annotation file ===
    txt_path = os.path.join(yolo_dir, os.path.splitext(img_name)[0] + ".txt")

    ch, cw = cropped.shape[:2]
    with open(txt_path, "w") as f:
        for cnt in contours:
            shifted_cnt = cnt - [x, y]  # shift contour into cropped space
            polygon = []
            for point in shifted_cnt:
                px, py = point[0]
                nx = px / cw
                ny = py / ch
                polygon.extend([nx, ny])

            if len(polygon) < 6:  # skip invalid polygons
                continue

            class_id = 0
            polygon_str = " ".join([f"{p:.6f}" for p in polygon])
            f.write(f"{class_id} {polygon_str}\n")

    return result