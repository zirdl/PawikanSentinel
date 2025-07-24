#!/usr/bin/env python
"""
This script combines the processed GTST-2023 and SeaTurtleID2022 datasets
and creates the dataset.yaml file for YOLOv5.
"""

import argparse
import os
import shutil
import yaml

def combine_datasets(gtst_dir, seaturtle_dir, output_dir):
    """Combines the two datasets."""
    for dataset_type in ["train", "val"]:
        for data_type in ["images", "labels"]:
            output_path = os.path.join(output_dir, data_type, dataset_type)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Copy GTST data
            gtst_path = os.path.join(gtst_dir, data_type, dataset_type)
            for f in os.listdir(gtst_path):
                shutil.copy(os.path.join(gtst_path, f), output_path)

            # Copy SeaTurtleID data
            seaturtle_path = os.path.join(seaturtle_dir, data_type, dataset_type)
            for f in os.listdir(seaturtle_path):
                shutil.copy(os.path.join(seaturtle_path, f), output_path)

def create_dataset_yaml(output_dir):
    """Creates the dataset.yaml file."""
    data = {
        'train': os.path.join(output_dir, 'images', 'train'),
        'val': os.path.join(output_dir, 'images', 'val'),
        'nc': 1,
        'names': ['turtle']
    }

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(data, f)

def main():
    parser = argparse.ArgumentParser(description="Combine datasets for YOLOv5.")
    parser.add_argument("--gtst_dir", default="datasets/processed/gtst-2023", help="Path to the processed GTST-2023 directory.")
    parser.add_argument("--seaturtle_dir", default="datasets/processed/seaturtleid2022", help="Path to the processed SeaTurtleID2022 directory.")
    parser.add_argument("--output_dir", default="datasets/combined", help="Path to the output directory.")
    args = parser.parse_args()

    combine_datasets(args.gtst_dir, args.seaturtle_dir, args.output_dir)
    create_dataset_yaml(args.output_dir)

    print("Datasets combined successfully.")

if __name__ == "__main__":
    main()
