import ujson as json
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from . import _utils
import os
from project.utils import sort_nicely


def _add_backward(data):
    data['ts_data']['backward'] = data['ts_data']['forward']
    # backward values
    data['ts_data']['backward']['values'] = data['ts_data']['backward']['values'][::-1]
    data['ts_data']['backward']['masks'] = data['ts_data']['backward']['masks'][::-1]
    data['ts_data']['backward']['deltas'] = data['ts_data']['backward']['deltas'][::-1]

    return data


def _parse_delta(masks, dir_):
    masks = np.array(masks)
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []
    seq_len = len(masks)
    data_dim = len(masks[0])

    for month in range(seq_len):
        if month == 0:
            deltas.append(np.ones(data_dim))
        else:
            deltas.append(np.ones(data_dim) + (1 - masks[month - 1]) * deltas[-1])

    return np.array(deltas)


class LULCInference(Dataset):
    def __init__(self, json_dir: str, level: str = 'N1', transform=None, label: int = 0):

        self.json_dir = json_dir
        json_path_pattern = os.path.join(self.json_dir, '*.json')

        self.json_paths = sorted(glob.glob(json_path_pattern))

        self.transform = transform
        self.label = label

        self.classes, self.class_to_idx, self.idx_to_class = _utils.get_classes_by_level(level)

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        with open(self.json_paths[idx], 'r') as json_file:
            data = json.load(json_file)

        if self.transform is not None:
            data = self.transform(data)

        rec = {
            'ts_data': {},
            'ancillary_data': data['ancillary_data'],
            'json_path': self.json_paths[idx],
            'label': self.label
        }

        if 'square_id' in data['metadata'].keys():
            rec['id'] = data['metadata']['square_id']
        else:
            rec['id'] = self.json_paths[idx].split('/')[-1].split('.')[0]

        rec['ts_data']['forward'] = data['ts_data']

        if 'probs' not in rec.keys():
            probs = [0]
            rec['probs'] = probs

        # Add backward direction
        rec = _add_backward(rec)

        # Fix deltas
        rec['ts_data']['forward']['deltas'] = _parse_delta(rec['ts_data']['forward']['masks'], 'forward')
        rec['ts_data']['backward']['deltas'] = _parse_delta(rec['ts_data']['backward']['masks'], 'backward')

        return rec

    def get_class_vars(self):
        return self.classes, self.class_to_idx, self.idx_to_class

