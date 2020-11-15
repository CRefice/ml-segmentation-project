import tifffile
from pathlib import Path
import numpy as np
import torch


def normalize(array: np.ndarray) -> np.ndarray:
    """ Normalize integer data to be a float between 0-1. """
    return array / np.iinfo(array.dtype).max


def to_tensor(array: np.ndarray) -> torch.tensor:
    # The nn module wants dimensions in the order (item, channel, height, width).
    # We only get item, height and width, so we add an extra channel dimension.
    return torch.from_numpy(array[:, None, :, :])


def load_dataset(name: str, threshold=False, data_path: Path = Path("dataset")):
    """
    Return two tensors, one containing the input data,
    the other the ground truth labels.

    Arguments:
    name -- the name of the dataset to load (the common part of their file names)
    threshold -- if true, do not consider the different identities of the ground truth,
      but just threshold it to background or foreground. (default False)
    data_path -- the path to the root folder of the dataset. (default "dataset/")
    """
    input_sequence = tiff.TiffSequence(data_path / "rawimages" / (name + "*.tif"))
    label_sequence = tiff.TiffSequence(data_path / "groundtruth" / (name + "*.tif"))
    data = to_tensor(normalize(input_sequence.asarray()))
    labels = to_tensor(label_sequence.asarray().astype(np.float32))
    if threshold:
        labels[labels > 0] = 1.0
    return (data, labels)
