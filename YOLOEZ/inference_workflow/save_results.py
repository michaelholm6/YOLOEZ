import sys
import os
import cv2
import json
from PyQt5 import QtWidgets


def get_save_path(default_name="inference_results"):
    """
    Prompt the user to select a directory for saving inference results.
    
    Args:
        default_name (str): Default suggested directory name.
    
    Returns:
        str or None: The chosen directory path, or None if cancelled.
    """
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        owns_app = True

    dialog = QtWidgets.QFileDialog()
    dialog.setWindowTitle("Select Directory to Save Inference Results")
    dialog.setFileMode(QtWidgets.QFileDialog.Directory)
    dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
    dialog.selectFile(default_name)

    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        save_dir = dialog.selectedFiles()[0]
    else:
        save_dir = None

    if owns_app:
        app.quit()

    return save_dir


def postprocess_and_save_results(results, original_images, save_dir=None, save_images=True, save_json=True, print_summary=True):
    """
    Postprocess YOLO inference results and save annotated images + JSON.

    Args:
        results (list): List of Ultralytics YOLO Result objects.
        save_dir (str, optional): Directory to save results. If None, prompts the user.
        save_images (bool): Whether to save annotated images.
        save_json (bool): Whether to save JSON summary of detections.
        print_summary (bool): Whether to print detection summary to console.

    Returns:
        list: Structured list of detections per image.
    """
    if save_dir is None:
        save_dir = get_save_path()
        if save_dir is None:
            print("No save directory selected. Skipping save.")
            save_images = save_json = False

    os.makedirs(save_dir, exist_ok=True)
    all_detections = []
    detections_summary = []

    for i, result in enumerate(results):
        img_detections = []
        boxes = result.boxes
        names = result.names
        masks = getattr(result, "masks", None)  # YOLOv8 segmentation masks

        if print_summary:
            print(f"\nImage {i+1}: {len(boxes)} detections")

        for j, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            label = names[cls_id] if names and cls_id in names else str(cls_id)

            polygon = None
            if masks:
                mask_data = masks.data[j].cpu().numpy()  # binary mask
                contours, _ = cv2.findContours(mask_data.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    polygon = contours[0].reshape(-1, 2).tolist()  # first contour as list of [x, y]

            det = {
                "image_index": i + 1,
                "class_id": cls_id,
                "class_name": label,
                "confidence": conf,
                "bbox": xyxy,
                "segmentation": polygon
            }
            img_detections.append(det)
            all_detections.append(det)

            if print_summary:
                print(f" - {label} ({conf:.2f}) at {xyxy}")
                if polygon:
                    print(f"   Segmentation polygon: {polygon}")

        detections_summary.append(img_detections)

        # Save annotated images
        if save_images and original_images is not None:
            # Use the original RGB image for plotting instead of the preprocessed grayscale
            annotated_img = result.plot(img=original_images[i])  # YOLOv8 accepts img= for plotting
            img_path = os.path.join(save_dir, f"result_{i+1}.jpg")
            cv2.imwrite(img_path, annotated_img)
            if print_summary:
                print(f"Saved annotated image to {img_path}")

    # Save JSON summary
    if save_json:
        json_path = os.path.join(save_dir, "detections.json")
        with open(json_path, "w") as f:
            json.dump(all_detections, f, indent=4)
        if print_summary:
            print(f"Saved JSON summary to {json_path}")

