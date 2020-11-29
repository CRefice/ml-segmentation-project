import tifffile as tiff
from pathlib import Path
import torch
from torch.utils.data import Dataset


class CellSegmentationDataset(Dataset):
    def __init__(
        self,
        raw_img_dir: Path,
        ground_truth_dir: Path,
        pattern: str = "",
        transform=None,
        target_transform=None,
    ):
        """
        Args:
            raw_image_dir: Directory with all the input images.
            ground_truth_dir: Directory with all the already-segmented output images.
            pattern: the pattern images must satisfy to be part of the dataset. leave blank to match all images
            transform (callable, optional): Optional transform to be applied on an image sample.
            target_transform (callable, optional): Optional transform to be applied on a target (segmentation) sample.
        """
        pattern += "*.tif"
        self.raw_img_names = sorted(raw_img_dir.glob(pattern))
        self.ground_truth_names = sorted(ground_truth_dir.glob(pattern))
        assert len(self.raw_img_names) == len(self.ground_truth_names)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.raw_img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = tiff.imread(self.raw_img_names[idx])
        if image.ndim > 2:
            # Retrieve only one channel in case image is RGB
            image = image[..., 0].squeeze()

        if self.transform:
            image = self.transform(image)

        segmentation = tiff.imread(self.ground_truth_names[idx])
        if self.target_transform:
            segmentation = self.target_transform(segmentation)

        return (image, segmentation)
