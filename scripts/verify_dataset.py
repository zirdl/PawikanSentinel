import argparse
import os
import yaml
from pathlib import Path
from tqdm import tqdm

def print_header(title):
    """Prints a formatted header."""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def print_status(check, success, message):
    """Prints a check status with a colored icon."""
    icon = "✅" if success else "❌"
    print(f"  {icon} {check:<50} {message}")

def verify_dataset_yaml(dataset_dir):
    """Verifies the dataset.yaml file."""
    print_header("Checking dataset.yaml")
    yaml_path = Path(dataset_dir) / 'dataset.yaml'
    
    if not yaml_path.exists():
        print_status("dataset.yaml found", False, f"File not found at {yaml_path}")
        return None, False

    print_status("dataset.yaml found", True, f"File found at {yaml_path}")

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print_status("YAML parsing", False, f"Error parsing YAML: {e}")
        return None, False
    
    print_status("YAML parsing", True, "Successfully parsed")

    checks = {
        'train': ('path', str),
        'val': ('path', str),
        'nc': ('number of classes', int),
        'names': ('class names', list)
    }
    
    all_checks_ok = True
    for key, (desc, dtype) in checks.items():
        if key not in data:
            print_status(f"'{key}' key exists", False, f"Missing required key")
            all_checks_ok = False
        elif not isinstance(data[key], dtype):
            print_status(f"'{key}' key type", False, f"Expected {dtype}, got {type(data[key])}")
            all_checks_ok = False
        else:
            print_status(f"'{key}' key valid", True, "Key present and type is correct")

    if 'nc' in data and 'names' in data and isinstance(data['nc'], int) and isinstance(data['names'], list):
        if len(data['names']) != data['nc']:
            print_status("Class count match", False, f"nc={data['nc']}, but found {len(data['names'])} names")
            all_checks_ok = False
        else:
            print_status("Class count match", True, "Number of classes matches names list")

    return data, all_checks_ok

def verify_directory_structure(dataset_dir, config):
    """Verifies the directory structure based on dataset.yaml."""
    print_header("Checking Directory Structure")
    
    if not config:
        print("  Skipping directory structure check due to missing config.")
        return False

    all_paths_exist = True
    for split in ['train', 'val']:
        path_key = config.get(split)
        if not path_key:
            continue

        image_path = Path(dataset_dir) / path_key
        label_path = Path(str(image_path).replace('images', 'labels'))

        if not image_path.exists() or not image_path.is_dir():
            print_status(f"{split} image directory", False, f"Not found at {image_path}")
            all_paths_exist = False
        else:
            print_status(f"{split} image directory", True, f"Found at {image_path}")

        if not label_path.exists() or not label_path.is_dir():
            print_status(f"{split} label directory", False, f"Not found at {label_path}")
            all_paths_exist = False
        else:
            print_status(f"{split} label directory", True, f"Found at {label_path}")
            
    return all_paths_exist

def verify_file_consistency(dataset_dir, config):
    """Verifies that image and label files are consistent."""
    print_header("Checking File Consistency")
    if not config:
        print("  Skipping file consistency check due to missing config.")
        return False

    overall_consistent = True
    for split in ['train', 'val']:
        path_key = config.get(split)
        if not path_key:
            continue

        image_path = Path(dataset_dir) / path_key
        label_path = Path(str(image_path).replace('images', 'labels'))

        if not image_path.exists() or not label_path.exists():
            continue

        image_files = {p.stem for p in image_path.glob('*.*')}
        label_files = {p.stem for p in label_path.glob('*.txt')}

        num_images = len(image_files)
        num_labels = len(label_files)

        if num_images != num_labels:
            print_status(f"{split} file count", False, f"Found {num_images} images and {num_labels} labels")
            overall_consistent = False
        else:
            print_status(f"{split} file count", True, f"Found {num_images} images and {num_labels} labels")

        unmatched_labels = label_files - image_files
        if unmatched_labels:
            print_status(f"{split} unmatched labels", False, f"{len(unmatched_labels)} labels without images")
            if len(unmatched_labels) < 5:
                for label in unmatched_labels:
                    print(f"    - {label}.txt")
            overall_consistent = False
        
        unmatched_images = image_files - label_files
        if unmatched_images:
            print_status(f"{split} unmatched images", False, f"{len(unmatched_images)} images without labels")
            if len(unmatched_images) < 5:
                for image in unmatched_images:
                    print(f"    - {image}")
            overall_consistent = False

        if not unmatched_labels and not unmatched_images and num_images > 0:
            print_status(f"{split} image-label pairs", True, "All images have corresponding labels")

    return overall_consistent

def verify_label_format(dataset_dir, config):
    """Verifies the format of a sample of label files."""
    print_header("Checking Label Format (Sample)")
    if not config or 'nc' not in config:
        print("  Skipping label format check due to missing config.")
        return False

    nc = config['nc']
    overall_valid = True
    files_to_check = 5

    for split in ['train', 'val']:
        path_key = config.get(split)
        if not path_key:
            continue
        
        image_path = Path(dataset_dir) / path_key
        label_path = Path(str(image_path).replace('images', 'labels'))

        if not label_path.exists():
            continue

        label_files = list(label_path.glob('*.txt'))
        if not label_files:
            continue

        print(f"\n  Checking {split} labels...")
        sample_files = label_files[:files_to_check]
        
        for file in tqdm(sample_files, desc=f"  Scanning {split} labels", unit="file"):
            with open(file, 'r') as f:
                for i, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print_status(f"{file.name} L{i}", False, f"Expected 5 values, got {len(parts)}")
                        overall_valid = False
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        coords = [float(p) for p in parts[1:]]

                        if not (0 <= class_id < nc):
                            print_status(f"{file.name} L{i}", False, f"Class ID {class_id} out of range [0, {nc-1}]")
                            overall_valid = False
                        
                        if not all(0.0 <= c <= 1.0 for c in coords):
                            print_status(f"{file.name} L{i}", False, "Coordinates are not normalized (0-1)")
                            overall_valid = False

                    except ValueError:
                        print_status(f"{file.name} L{i}", False, "Contains non-numeric values")
                        overall_valid = False
    
    if overall_valid:
        print_status("Sampled label format", True, "All checked labels seem valid")
    else:
        print_status("Sampled label format", False, "Found issues in label files")

    return overall_valid

def main():
    """Main function to run the verification script."""
    parser = argparse.ArgumentParser(description="Verify a YOLO dataset for training readiness.")
    parser.add_argument("dir", nargs='?', default=None, help="Path to the dataset directory.")
    args = parser.parse_args()

    dataset_dir = args.dir
    if not dataset_dir:
        dataset_dir = input("Enter the path to your dataset directory: ")

    if not os.path.isdir(dataset_dir):
        print(f"Error: Directory not found at '{dataset_dir}'")
        return

    print(f"\nStarting verification for dataset at: {dataset_dir}")

    config, yaml_ok = verify_dataset_yaml(dataset_dir)
    structure_ok = verify_directory_structure(dataset_dir, config)
    consistency_ok = verify_file_consistency(dataset_dir, config)
    format_ok = verify_label_format(dataset_dir, config)

    print_header("Final Summary")
    is_ready = yaml_ok and structure_ok and consistency_ok and format_ok
    
    if is_ready:
        print("🎉 Your dataset appears to be correctly formatted and ready for training! 🎉")
    else:
        print("⚠️  Your dataset has issues that need to be addressed before training. ⚠️")
        print("Please review the checks above for details on what to fix.")

if __name__ == "__main__":
    main()

