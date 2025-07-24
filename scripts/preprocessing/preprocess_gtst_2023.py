#!/usr/bin/env python
"""
This script preprocesses the GTST-2023 dataset for YOLOv5 training.

It performs the following steps:
1.  Extracts frames from the video files.
2.  Converts the annotations from the provided format to YOLO format.
3.  Splits the data into training and validation sets.
"""

import argparse
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_frames(video_path, output_dir):
    """Extracts frames from a video file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()

def convert_annotations(annotations_path, images_dir, labels_dir):
    """Converts annotations to YOLO format."""
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    annotations = pd.read_csv(annotations_path)

    for _, row in annotations.iterrows():
        image_filename = row['filename']
        image_path = os.path.join(images_dir, image_filename)
        label_path = os.path.join(labels_dir, os.path.splitext(image_filename)[0] + ".txt")

        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        height, width, _ = img.shape

        # Assuming the CSV has columns 'xmin', 'ymin', 'xmax', 'ymax'
        x_center = (row['xmin'] + row['xmax']) / 2 / width
        y_center = (row['ymin'] + row['ymax']) / 2 / height
        bbox_width = (row['xmax'] - row['xmin']) / width
        bbox_height = (row['ymax'] - row['ymin']) / height

        with open(label_path, 'w') as f:
            f.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")

def main():
    parser = argparse.ArgumentParser(description="Preprocess the GTST-2023 dataset.")
    parser.add_argument("--dataset_dir", default="datasets/gtst-2023", help="Path to the GTST-2023 dataset directory.")
    parser.add_argument("--output_dir", default="datasets/processed/gtst-2023", help="Path to the output directory.")
    parser.add_argument("--val_size", type=float, default=0.2, help="Size of the validation set.")
    args = parser.parse_args()

    # This is a placeholder. The actual paths to videos and annotations need to be determined
    # by inspecting the downloaded dataset.
    video_dir = os.path.join(args.dataset_dir, "videos")
    annotations_path = os.path.join(args.dataset_dir, "annotations.csv")

    images_dir = os.path.join(args.output_dir, "images")
    labels_dir = os.path.join(args.output_dir, "labels")

    # Extract frames from videos
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        frame_output_dir = os.path.join(images_dir, video_name)
        extract_frames(video_path, frame_output_dir)

    # Convert annotations
    convert_annotations(annotations_path, images_dir, labels_dir)

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

    print("GTST-2023 dataset preprocessing complete.")

if __name__ == "__main__":
    main()
