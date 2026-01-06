import os
import cv2
import numpy as np

def save_box_results(
    image_paths,
    boxes_dict,
    line_thickness,
    output_dir,
    areas_of_interest,
    save_yolo_dataset=True,
    save_unlabeled_images=False
):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Folder for box visualizations
    box_vis_dir = os.path.join(output_dir, "box_visualizations")
    os.makedirs(box_vis_dir, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}: failed to load.")
            continue

        boxes = boxes_dict.get(img_path, [])
        boxes = [np.array(b, dtype=np.int32) for b in boxes]

        # 🔹 Only skip if unlabeled images are NOT allowed
        if len(boxes) == 0 and not save_unlabeled_images:
            print(f"{img_path}: no boxes → skipping sample.")
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

        # ---------- Draw boxes ----------
        box_vis = base_img.copy()
        for box in boxes:
            pts = box.reshape(-1, 2)

            bx1 = np.min(pts[:, 0])
            by1 = np.min(pts[:, 1])
            bx2 = np.max(pts[:, 0])
            by2 = np.max(pts[:, 1])

            bx1 = int(np.clip(bx1, 0, w-1))
            bx2 = int(np.clip(bx2, 0, w-1))
            by1 = int(np.clip(by1, 0, h-1))
            by2 = int(np.clip(by2, 0, h-1))

            cv2.rectangle(
                box_vis,
                (bx1, by1),
                (bx2, by2),
                (0, 255, 0),
                int(line_thickness)
            )

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # ---------- Save box visualization ----------
        box_img_path = os.path.join(
            box_vis_dir,
            f"{img_name}_boxes.png"
        )
        cv2.imwrite(box_img_path, box_vis)

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
                for box in boxes:
                    pts = box.reshape(-1, 2)

                    bx1 = np.min(pts[:, 0])
                    by1 = np.min(pts[:, 1])
                    bx2 = np.max(pts[:, 0])
                    by2 = np.max(pts[:, 1])

                    bx1 = int(np.clip(bx1, 0, w_img-1))
                    bx2 = int(np.clip(bx2, 0, w_img-1))
                    by1 = int(np.clip(by1, 0, h_img-1))
                    by2 = int(np.clip(by2, 0, h_img-1))

                    bw = bx2 - bx1
                    bh = by2 - by1
                    if bw <= 0 or bh <= 0:
                        continue

                    cx = (bx1 + bx2) / 2 / w_img
                    cy = (by1 + by2) / 2 / h_img

                    class_id = 0
                    f.write(
                        f"{class_id} {cx:.6f} {cy:.6f} {bw / w_img:.6f} {bh / h_img:.6f}\n"
                    )

        results.append(box_vis)

    return results
