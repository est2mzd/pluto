#!/usr/bin/env python3
"""
Create nuplan_boston.yaml splitter configuration by randomly splitting train_boston files.

Usage:
    python create_nuplan_boston_splitter.py [train_ratio] [val_ratio] [test_ratio] [output_file] [seed]
    
Examples:
    python create_nuplan_boston_splitter.py  # Uses defaults (70% train, 15% val, 15% test)
    python create_nuplan_boston_splitter.py 0.6 0.2 0.2  # 60% train, 20% val, 20% test
    python create_nuplan_boston_splitter.py 0.7 0.15 0.15 /custom/path/nuplan_boston.yaml 42  # With custom path and seed
"""

import os
import random
import sys
from pathlib import Path
from typing import Tuple, List

# Configuration
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
BOSTON_DATASET_PATH = "/nuplan/dataset/nuplan-v1.1/splits/train_boston/"
DEFAULT_OUTPUT_FILE = "/workspace/nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan_boston.yaml"


def get_db_files(dataset_path: str) -> List[str]:
    """Get all .db files from the dataset directory, sorted."""
    files = []
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    for file in os.listdir(dataset_path):
        if file.endswith(".db"):
            # Remove .db extension to get the log name
            log_name = file[:-3]
            files.append(log_name)
    
    if not files:
        raise ValueError(f"No .db files found in {dataset_path}")
    
    return sorted(files)


def split_files(files: List[str], train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Randomly split files into train, validation, and test sets.
    
    Args:
        files: List of file names
        train_ratio: Ratio for training set (0.0 to 1.0)
        val_ratio: Ratio for validation set (0.0 to 1.0)
        test_ratio: Ratio for test set (0.0 to 1.0)
        seed: Random seed for reproducibility (None for random)
    
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:  # Allow small floating point errors
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    if seed is not None:
        random.seed(seed)
    
    # Create a shuffled copy
    shuffled = files.copy()
    random.shuffle(shuffled)
    
    # Calculate split points
    train_idx = int(len(shuffled) * train_ratio)
    val_idx = train_idx + int(len(shuffled) * val_ratio)
    
    train = shuffled[:train_idx]
    val = shuffled[train_idx:val_idx]
    test = shuffled[val_idx:]
    
    return sorted(train), sorted(val), sorted(test)


def generate_yaml(train_files: List[str], val_files: List[str], test_files: List[str]) -> str:
    """Generate YAML content in the format of nuplan.yaml."""
    yaml_content = "_target_: nuplan.planning.training.data_loader.log_splitter.LogSplitter\n"
    yaml_content += "_convert_: 'all'\n\n"
    yaml_content += "log_splits:\n"
    
    # Train section
    yaml_content += "  train:\n"
    for file in train_files:
        yaml_content += f"    - {file}\n"
    
    # Val section
    yaml_content += "  val:\n"
    for file in val_files:
        yaml_content += f"    - {file}\n"
    
    # Test section
    yaml_content += "  test:\n"
    for file in test_files:
        yaml_content += f"    - {file}\n"
    
    return yaml_content


def main():
    """Main function."""
    # Parse arguments
    train_ratio = DEFAULT_TRAIN_RATIO
    val_ratio = DEFAULT_VAL_RATIO
    test_ratio = DEFAULT_TEST_RATIO
    output_file = DEFAULT_OUTPUT_FILE
    seed = None
    
    if len(sys.argv) > 1:
        try:
            train_ratio = float(sys.argv[1])
        except ValueError:
            print(f"Error: train_ratio must be a float, got '{sys.argv[1]}'")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            val_ratio = float(sys.argv[2])
        except ValueError:
            print(f"Error: val_ratio must be a float, got '{sys.argv[2]}'")
            sys.exit(1)
    
    if len(sys.argv) > 3:
        try:
            test_ratio = float(sys.argv[3])
        except ValueError:
            print(f"Error: test_ratio must be a float, got '{sys.argv[3]}'")
            sys.exit(1)
    
    if len(sys.argv) > 4:
        output_file = sys.argv[4]
    
    if len(sys.argv) > 5:
        try:
            seed = int(sys.argv[5])
        except ValueError:
            print(f"Error: seed must be an integer, got '{sys.argv[5]}'")
            sys.exit(1)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: Ratios must sum to 1.0, got {total_ratio}")
        sys.exit(1)
    
    try:
        print(f"Reading files from: {BOSTON_DATASET_PATH}")
        files = get_db_files(BOSTON_DATASET_PATH)
        print(f"Found {len(files)} files")
        
        print(f"\nSplitting with ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
        if seed is not None:
            print(f"Using seed: {seed}")
        
        train_files, val_files, test_files = split_files(files, train_ratio, val_ratio, test_ratio, seed)
        
        print(f"Train set: {len(train_files)} files ({len(train_files)/len(files)*100:.1f}%)")
        print(f"Val set: {len(val_files)} files ({len(val_files)/len(files)*100:.1f}%)")
        print(f"Test set: {len(test_files)} files ({len(test_files)/len(files)*100:.1f}%)")
        
        # Generate YAML
        yaml_content = generate_yaml(train_files, val_files, test_files)
        
        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"\nSuccessfully created: {output_file}")
        print(f"File size: {len(yaml_content) / 1024:.1f} KB")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
