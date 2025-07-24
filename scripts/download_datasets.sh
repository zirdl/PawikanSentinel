#!/bin/bash

# This script downloads the GTST-2023 and SeaTurtleID2022 datasets from Kaggle.
# It requires the Kaggle API to be installed and configured with your credentials.

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to download a dataset from Kaggle
download_dataset() {
    local dataset_name="$1"
    local dataset_path="$2"

    echo "Downloading $dataset_name to $dataset_path..."

    if [ -d "$dataset_path" ]; then
        echo "Dataset already downloaded to $dataset_path. Skipping."
    else
        kaggle datasets download -d "$dataset_name" -p "$dataset_path" --unzip
        echo "Successfully downloaded and unzipped $dataset_name."
    fi
}

# Create a directory to store the datasets
mkdir -p datasets

# Download the datasets
download_dataset "irfanhipiny/gtst-2023" "datasets/gtst-2023"
download_dataset "wildlifedatasets/seaturtleid2022" "datasets/seaturtleid2022"

echo "All datasets downloaded successfully."
