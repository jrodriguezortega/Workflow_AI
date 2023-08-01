import pandas as pd

import numpy as np
import os
from glob import glob

from project.utils import sort_nicely

# Level 1 dicts
class_to_idx_N1 = {
    'Artificial': 0,
    'Agricultural': 1,
    'Terrestrial': 2,
    'Wetlands': 3,
}

idx_to_class_N1 = {v: k for k, v in class_to_idx_N1.items()}

# Level 2 dicts
class_to_idx_N2 = {
    'Artificial': 0,
    'Annual croplands': 1,
    'Greenhouses': 2,
    'Woody croplands': 3,
    'Combinations of croplands and natural vegetation': 4,
    'Grasslands': 5,
    'Shrubland': 6,
    'Forests': 7,
    'Barelands': 8,
    'Wetlands': 9,
}

idx_to_class_N2 = {v: k for k, v in class_to_idx_N2.items()}


def is_excluded(class_dir: str, excluded_classes: list):
    for excluded_class in excluded_classes:
        if excluded_class in class_dir:
            return True

    return False


def get_global_idxs(json_dir: str, excluded_classes: list):
    inter_part_dir = os.path.join(json_dir, 'partitions/')

    class_dirs = sort_nicely(glob(inter_part_dir + '*/'))

    num_samples = 0
    global_train_idxs = []
    global_test_idxs = []
    for class_dir in class_dirs:
        if is_excluded(class_dir, excluded_classes):
            continue

        class_name = class_dir.split('/')[-2]
        class_train_idxs = np.load(os.path.join(class_dir, 'train.npy'))
        class_test_idxs = np.load(os.path.join(class_dir, 'test.npy'))

        global_train_idxs.extend(class_train_idxs + num_samples)
        global_test_idxs.extend(class_test_idxs + num_samples)

        num_samples += len(class_train_idxs) + len(class_test_idxs)

    global_train_idxs = np.array(global_train_idxs)
    global_test_idxs = np.array(global_test_idxs)

    return global_train_idxs, global_test_idxs


def find_classes(directory: str):
    """Finds the class folders in a dataset.
    """
    classes = sort_nicely(list(entry.name for entry in os.scandir(directory) if entry.is_dir()))
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_classes = {i: cls_name for cls_name, i in class_to_idx.items()}

    return classes, class_to_idx, idx_to_classes


def find_classes_df(labels_df: pd.DataFrame):
    """ Finds the classes in the "label_df" """
    columns = sorted(labels_df.columns.tolist())
    classes = columns[:-2]

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_classes = {i: cls_name for cls_name, i in class_to_idx.items()}

    return classes, class_to_idx, idx_to_classes


def get_classes_by_level(level: str):
    classes = None
    class_to_idx = None
    idx_to_classes = None

    if level == 'N1':
        classes = list(class_to_idx_N1.keys())
        class_to_idx = class_to_idx_N1
        idx_to_classes = idx_to_class_N1
    elif level == 'N2':
        classes = list(class_to_idx_N2.keys())
        class_to_idx = class_to_idx_N2
        idx_to_classes = idx_to_class_N2

    return classes, class_to_idx, idx_to_classes


def get_square_id(json_path: str):
    square_id = int(json_path.split(sep='/')[-1].split(sep='.')[0])

    return square_id
