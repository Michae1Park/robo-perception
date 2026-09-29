"""Core vision library: detection (YOLOE), pose (FoundationPose), spatial (plane + occupancy), grasp."""
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    with open(os.path.join(REPO, 'config.yaml')) as f:
        return yaml.safe_load(f)
