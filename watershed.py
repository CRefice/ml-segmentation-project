import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def label_with_watershed(output_predictions, min_distance=50):
    """
    Creates an instance segmentation from a binary,
    foreground-background segmentation map using the watershed algorithm.

    Arguments:
      output_predictions -- Binary segmented image
      min_distance -- Minimum distance that cell-centers should be apart of eachother in pixels

    Returns:
      Labelled image with every cell having a different value (v > 0) and a border
      between every cell (1-pixel wide)
    """
    # Euclidean distance matrix of the image
    distance = ndi.distance_transform_edt(output_predictions)
    # List of (x,y) pairs representing the coordinates of the peaks
    coords_of_peaks = peak_local_max(
        distance,
        min_distance=min_distance,
        footprint=np.ones((64, 64)),
        labels=output_predictions,
    )
    # Black image with white pixels representing the peaks (cell-centers)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords_of_peaks.T)] = True
    # Mask transformed to having a different value per peak (cell-center)
    markers, _ = ndi.label(mask)

    # Return the watershed labels, use -distance because watershed works on minima.
    return watershed(-distance, markers, mask=output_predictions, watershed_line=True)
