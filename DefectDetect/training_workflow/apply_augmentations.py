import os
import cv2
import albumentations as A

def build_transform(aug_dict, task):
    """Build Albumentations transform from aug_dict flags."""
    transforms = []
    if aug_dict.get("flip"):
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug_dict.get("rotate"):
        transforms.append(A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5))
    if aug_dict.get("scale"):
        transforms.append(A.RandomScale(scale_limit=0.2, p=0.5))
    if aug_dict.get("color"):
        transforms.append(A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5))
    if aug_dict.get("blur"):
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))
    if aug_dict.get("noise"):
        transforms.append(A.GaussNoise(p=0.3))

    if task == "detection":
        return A.Compose(transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))
    else:  # segmentation
        return A.Compose(transforms, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

def detect_label_format(lbl_path):
    """
    Infer whether a YOLO txt file contains segmentation polygons or bbox coords.
    - Segmentation: more than 4 coords (x1 y1 x2 y2 ...).
    - Detection: exactly 4 coords after class_id.
    """
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            coords = parts[1:]
            if len(coords) > 4:
                return "segmentation"
            elif len(coords) == 4:
                return "detection"
    return None

def augment_yolo_dataset(dataset_dir, aug_dict, output_dir, num_augments=1, task="segmentation"):
    """
    Augment YOLO dataset (segmentation or detection) with Albumentations.

    Args:
        dataset_dir (str): Root YOLO dataset (images/train, labels/train, ...).
        aug_dict (dict): Augmentations to apply.
        output_dir (str): Where to save augmented dataset.
        num_augments (int): How many augmented copies per image.
        task (str): "segmentation" or "detection".
    """
    os.makedirs(output_dir, exist_ok=True)
    exts = [".jpg", ".jpeg", ".png", ".bmp"]

    for split in ["train", "val"]:
        img_in = os.path.join(dataset_dir, "images", split)
        lbl_in = os.path.join(dataset_dir, "labels", split)
        img_out = os.path.join(output_dir, "images", split)
        lbl_out = os.path.join(output_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for f in os.listdir(img_in):
            name, ext = os.path.splitext(f)
            if ext.lower() not in exts:
                continue

            img_path = os.path.join(img_in, f)
            lbl_path = os.path.join(lbl_in, name + ".txt")
            if not os.path.exists(lbl_path):
                continue

            # Detect format
            detected_format = detect_label_format(lbl_path)
            if detected_format != task:
                raise ValueError(f"Label format ({detected_format}) does not match selected task ({task}).")

            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            # Load labels
            labels = []
            with open(lbl_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    labels.append({"label": class_id, "coords": coords})

            # Save original image + labels
            cv2.imwrite(os.path.join(img_out, f"{name}{ext}"), img)
            with open(os.path.join(lbl_out, f"{name}.txt"), "w") as f_out:
                for lbl in labels:
                    if task == "segmentation":
                        f_out.write(f"{lbl['label']} " + " ".join(f"{c:.6f}" for c in lbl["coords"]) + "\n")
                    else:  # detection
                        f_out.write(f"{lbl['label']} " + " ".join(f"{c:.6f}" for c in lbl["coords"]) + "\n")

            # Prepare transform
            transform = build_transform(aug_dict, task)

            # Augment
            for aug_idx in range(num_augments):
                if task == "segmentation":
                    # Albumentations does not natively support YOLO polygons — treat as keypoints list
                    keypoints = [(lbl["coords"][i], lbl["coords"][i+1]) for lbl in labels for i in range(0, len(lbl["coords"]), 2)]
                    transformed = transform(image=img, keypoints=keypoints)
                    aug_img = transformed["image"]
                    aug_keypoints = transformed["keypoints"]

                    # regroup into polygons
                    coords = []
                    for (x, y) in aug_keypoints:
                        coords.extend([x / w, y / h])
                    aug_lbls = [{"label": lbl["label"], "coords": coords} for lbl in labels]

                else:  # detection
                    bboxes = []
                    class_labels = []
                    for lbl in labels:
                        x_center, y_center, bw, bh = lbl["coords"]
                        # convert YOLO to pascal_voc
                        x1 = (x_center - bw/2) * w
                        y1 = (y_center - bh/2) * h
                        x2 = (x_center + bw/2) * w
                        y2 = (y_center + bh/2) * h
                        bboxes.append([x1, y1, x2, y2])
                        class_labels.append(lbl["label"])

                    transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                    aug_img = transformed["image"]
                    aug_lbls = []
                    for bbox, cls in zip(transformed["bboxes"], transformed["class_labels"]):
                        x1, y1, x2, y2 = bbox
                        # convert back to YOLO
                        x_center = (x1 + x2) / 2 / w
                        y_center = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        aug_lbls.append({"label": cls, "coords": [x_center, y_center, bw, bh]})

                # Save augmented
                aug_img_name = f"{name}_aug{aug_idx}{ext}"
                cv2.imwrite(os.path.join(img_out, aug_img_name), aug_img)
                aug_lbl_path = os.path.join(lbl_out, f"{name}_aug{aug_idx}.txt")
                with open(aug_lbl_path, "w") as f_out:
                    for lbl in aug_lbls:
                        f_out.write(f"{lbl['label']} " + " ".join(f"{c:.6f}" for c in lbl["coords"]) + "\n")
