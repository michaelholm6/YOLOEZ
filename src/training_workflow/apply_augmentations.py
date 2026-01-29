# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import os
import cv2
import albumentations as A


# ---------------------------
# Transform builder
# ---------------------------
def build_transform(aug_dict, task):
    transforms = []

    if aug_dict.get("flip"):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug_dict.get("rotate") or aug_dict.get("scale"):
        transforms.append(
            A.Affine(
                rotate=(-30, 30) if aug_dict.get("rotate") else 0,
                scale=(0.8, 1.2) if aug_dict.get("scale") else 1.0,
                translate_percent=(0.0, 0.0),
                fit_output=False,
                p=0.5,
            )
        )

    if aug_dict.get("color"):
        transforms.append(A.ColorJitter(0.3, 0.3, 0.3, 0.1, p=0.5))

    if aug_dict.get("blur"):
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))

    if aug_dict.get("noise"):
        transforms.append(A.GaussNoise(p=0.3))

    # If no transforms are enabled, just return identity
    if not transforms:
        return A.NoOp()  # NoOp just returns the input image unchanged

    if task == "detection":
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                clip=True,
                min_visibility=0.1,
            ),
        )
    else:
        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )


# ---------------------------
# Label format detection
# ---------------------------
def detect_label_format(lbl_path):
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts[1:]) > 4:
                return "segmentation"
            elif len(parts[1:]) == 4:
                return "detection"
    return None


# ---------------------------
# Dataset augmentation
# ---------------------------
def augment_yolo_dataset(
    dataset_dir, aug_dict, output_dir, num_augments=1, task="segmentation"
):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for split in ["train", "val"]:
        img_in = os.path.join(dataset_dir, "images", split)
        lbl_in = os.path.join(dataset_dir, "labels", split)
        img_out = os.path.join(output_dir, "images", split)
        lbl_out = os.path.join(output_dir, "labels", split)

        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fname in os.listdir(img_in):
            name, ext = os.path.splitext(fname)
            if ext.lower() not in exts:
                continue

            img_path = os.path.join(img_in, fname)
            lbl_path = os.path.join(lbl_in, name + ".txt")
            if not os.path.exists(lbl_path):
                continue

            detected = detect_label_format(lbl_path)
            if detected != task and detected is not None:
                print(f"Skipping {name}: expected {task}, found {detected}")
                continue

            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            labels = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    labels.append(
                        {
                            "label": int(parts[0]),
                            "coords": [float(x) for x in parts[1:]],
                        }
                    )

            # Save original
            cv2.imwrite(os.path.join(img_out, fname), img)
            with open(os.path.join(lbl_out, name + ".txt"), "w") as f:
                for l in labels:
                    f.write(
                        f"{l['label']} "
                        + " ".join(f"{c:.6f}" for c in l["coords"])
                        + "\n"
                    )

            transform = build_transform(aug_dict, task)

            for k in range(num_augments):
                if task == "detection":
                    bboxes, class_labels = [], []

                    for l in labels:
                        xc, yc, bw, bh = l["coords"]
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        x2 = (xc + bw / 2) * w
                        y2 = (yc + bh / 2) * h
                        bboxes.append([x1, y1, x2, y2])
                        class_labels.append(l["label"])

                    out = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                    aug_img = out["image"]
                    aug_h, aug_w = aug_img.shape[:2]

                    aug_labels = []
                    for (x1, y1, x2, y2), cls in zip(
                        out["bboxes"], out["class_labels"]
                    ):
                        xc = ((x1 + x2) / 2) / aug_w
                        yc = ((y1 + y2) / 2) / aug_h
                        bw = (x2 - x1) / aug_w
                        bh = (y2 - y1) / aug_h
                        aug_labels.append({"label": cls, "coords": [xc, yc, bw, bh]})

                else:  # segmentation
                    polygons = []
                    for l in labels:
                        pts = [
                            (l["coords"][i] * w, l["coords"][i + 1] * h)
                            for i in range(0, len(l["coords"]), 2)
                        ]
                        polygons.append(pts)

                    flat_pts = [pt for poly in polygons for pt in poly]
                    out = transform(image=img, keypoints=flat_pts)
                    aug_img = out["image"]
                    aug_h, aug_w = aug_img.shape[:2]

                    aug_labels = []
                    idx = 0
                    for l, poly in zip(labels, polygons):
                        n = len(poly)
                        pts = out["keypoints"][idx : idx + n]
                        idx += n

                        coords = []
                        for x, y in pts:
                            coords.extend([x / aug_w, y / aug_h])

                        aug_labels.append({"label": l["label"], "coords": coords})

                # Save augmented
                aug_img_name = f"{name}_aug{k}{ext}"
                cv2.imwrite(os.path.join(img_out, aug_img_name), aug_img)

                with open(os.path.join(lbl_out, f"{name}_aug{k}.txt"), "w") as f:
                    for l in aug_labels:
                        f.write(
                            f"{l['label']} "
                            + " ".join(f"{c:.6f}" for c in l["coords"])
                            + "\n"
                        )
