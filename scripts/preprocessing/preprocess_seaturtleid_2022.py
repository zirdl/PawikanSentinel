#!/usr/bin/env python
"""
This script preprocesses the SeaTurtleID2022 dataset for YOLOv5 training.

It performs the following steps:
1.  Converts the annotations from the provided format to YOLO format.
2.  Splits the data into training and validation sets.
"""

import argparse
import os
import json
from sklearn.model_selection import train_test_split

def convert_coco_to_yolo(coco_annotations_path, images_dir, labels_dir):
    """Converts COCO annotations to YOLO format."""
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    images = {image['id']: image for image in coco_data['images']}
    annotations = coco_data['annotations']

    for ann in annotations:
        image_id = ann['image_id']
        image_info = images[image_id]
        image_width = image_info['width']
        image_height = image_info['height']

        # COCO bbox is [x, y, width, height]
        x, y, w, h = ann['bbox']
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        bbox_width = w / image_width
        bbox_height = h / image_height

        image_filename = image_info['file_name']
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)

        with open(label_path, 'a') as f:
            f.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")

def main():
    parser = argparse.ArgumentParser(description="Preprocess the SeaTurtleID2022 dataset.")
    parser.add_argument("--dataset_dir", default="datasets/seaturtleid2022", help="Path to the SeaTurtleID2022 dataset directory.")
    parser.add_argument("--output_dir", default="datasets/processed/seaturtleid2022", help="Path to the output directory.")
    parser.add_argument("--val_size", type=float, default=0.2, help="Size of the validation set.")
    args = parser.parse_args()

    # This is a placeholder. The actual paths to annotations need to be determined
    # by inspecting the downloaded dataset.
    annotations_path = os.path.join(args.dataset_dir, "annotations.json")
    images_dir = os.path.join(args.dataset_dir, "images")

    labels_dir = os.path.join(args.output_dir, "labels")

    # Convert annotations
    convert_coco_to_yolo(annotations_path, images_dir, labels_dir)

    # Split data into train and validation sets
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    train_files, val_files = train_test_split(image_files, test_size=args.val_size, random_state=42)

    for dataset in [("train", train_files), ("val", val_files)]:
        dataset_name, files = dataset
        dataset_images_dir = os.path.join(args.output_dir, "images", dataset_name)
        dataset_labels_dir = os.path.join(args.output_dir, "labels", dataset_name)

        if not os.path.exists(dataset_images_dir):
            os.makedirs(dataset_images_dir)
        if not os.path.exists(dataset_labels_dir):
            os.makedirs(dataset_labels_dir)

        for f in files:
            os.rename(os.path.join(images_dir, f), os.path.join(dataset_images_dir, f))
            label_file = os.path.splitext(f)[0] + ".txt"
            os.rename(os.path.join(labels_dir, label_file), os.path.join(dataset_labels_dir, label_file))

    print("SeaTurtleID2022 dataset preprocessing complete.")

if __name__ == "__main__":
    main()
