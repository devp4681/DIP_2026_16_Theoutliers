"""
AID Multi-Label Dataset Loader
Handles:
- Loading multilabel .csv annotations (3,000 images, 17 classes)
- Train/Validation/Test split mappings
- Image path resolution
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image

LABEL_NAMES = (
    'airplane', 'bare-soil', 'buildings', 'cars', 'chaparral',
    'court', 'dock', 'field', 'grass', 'mobile-home',
    'pavement', 'sand', 'sea', 'ship', 'tanks', 'trees', 'water'
)


def load_annotations(csv_path: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load multilabel annotations from CSV file.
    Returns:
        labels_dict: mapping from image stem/filename to binary numpy array of length 17
        classes: list of class names
    """
    df = pd.read_csv(csv_path)
    img_col = df.columns[0]
    classes = [c.strip() for c in df.columns[1:]]
    
    labels_dict = {}
    for _, row in df.iterrows():
        img_name = str(row[img_col]).strip()
        vec = row[classes].values.astype(int)
        labels_dict[img_name] = vec
        
    return labels_dict, classes


def load_splits(split_json_path: Path) -> dict:
    """
    Load precomputed reference/validation split indices from JSON.
    """
    if not split_json_path.exists():
        return {}
    with open(split_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_load_image(image_path: Path) -> np.ndarray:
    """
    Safely load an image using PIL to avoid libjpeg truncation issues,
    converting to RGB numpy array.
    """
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def parse_image_name(image_path: Path) -> Tuple[str, str]:
    """
    Extract image stem and scene class name from image path.
    Example: 'images_tr/airport_10.jpg' -> ('airport_10', 'airport')
    """
    stem = image_path.stem
    scene_class = stem.rsplit('_', 1)[0]
    return stem, scene_class
