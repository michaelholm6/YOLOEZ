# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import yaml
import os

def create_yaml(dataset_path):
    yaml_file = os.path.join(dataset_path, "dataset.yaml")
    yaml_dict = {
        'path': os.path.abspath(dataset_path),
        'train': 'augmented_dataset/images/train',
        'val': 'augmented_dataset/images/val',
        'nc': 1,
        'names': ['detection']
    }
    
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_dict, f)
        
    return yaml_file