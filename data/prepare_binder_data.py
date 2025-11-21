#!/usr/bin/env python3
import os
import argparse
import yaml
import subprocess
from omegaconf import OmegaConf


def update_yaml_config(pdb_dir):
    """Update the csv_path field in the original YAML config"""
    # Load original config file
    config_path = "../configs/binder_design.yaml"
    config = OmegaConf.load(config_path)

    # Backup original path
    original_csv_path = config.data.dataset.csv_path

    # Update csv_path
    config.data.dataset.csv_path = os.path.join(pdb_dir, 'preprocessed/metadata.csv')

    # Save updated config
    OmegaConf.save(config, config_path)

    return original_csv_path


def prepare_binder_data(pdb_dir):
    """Prepare all necessary data for binder design"""
    # Ensure absolute path
    pdb_dir = os.path.abspath(pdb_dir)
    print(f"Processing PDB directory: {pdb_dir}")

    # Create preprocessed directory
    preprocessed_dir = os.path.join(pdb_dir, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Step 1: Run process_pdb_files.py
    print("Step 1: Processing PDB files...")
    process_cmd = [
        'python',
        'process_pdb_files.py',
        '--pdb_dir', pdb_dir
    ]
    subprocess.run(process_cmd, check=True)

    # Step 2: Update YAML config
    print("Step 2: Updating configuration file...")
    original_csv_path = update_yaml_config(pdb_dir)

    try:
        # Step 3: Run collect_pkl.py
        print("Step 3: Collecting data into pkl file...")
        collect_cmd = [
            'python',
            'collect_pkl.py'
        ]
        subprocess.run(collect_cmd, check=True)

        print("Data processing completed!")
        print(f"Output files are located at: {preprocessed_dir}")

    finally:
        # Restore original config
        config = OmegaConf.load("../configs/binder_design.yaml")
        config.data.dataset.csv_path = original_csv_path
        OmegaConf.save(config, "../configs/binder_design.yaml")


def main():
    parser = argparse.ArgumentParser(description='Prepare data required for binder design')
    parser.add_argument(
        '--pdb_dir',
        help='Path to the directory containing PDB files',
        default='//home/junyu/project/binder_target/ni2k/',
    )
    args = parser.parse_args()

    prepare_binder_data(args.pdb_dir)


if __name__ == '__main__':
    main()
