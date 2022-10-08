import os.path
import numpy as np
import torch

#import common.utils # Could make failures in pyrenderer bib
import pyrenderer


def get_tensor_from_cvol(filepath):
    if not os.path.exists(filepath):
        raise ValueError(f'[ERROR] Volume file {filepath} does not exist.')
    vol = pyrenderer.Volume(filepath)
    feature = vol.get_feature(0)
    level = feature.get_level(0).to_tensor()
    return level