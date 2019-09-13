# Patchify lib
import numpy as np
from skimage.util import view_as_windows
from itertools import product
from typing import Tuple


def patchify(patches: np.ndarray, patch_size: Tuple[int, int], step = 1):
    return view_as_windows(patches, patch_size, step)

