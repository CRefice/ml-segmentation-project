# Project structure

The project is split up into a python package, containing most of the code for the models, training, and other relevant implementation work,
and a Jupyter notebook, that will walk you through actually using our code to train/download a model and applying it to the dataset.

## Contents of the code files

- `dataset.py`: Data structures to load, transform and augment the nuclei segmentation dataset, which can be downloaded [through this link.](https://www.ebi.ac.uk/biostudies/files/S-BSST265/dataset.zip) or by running the `fetch-data.sh` script included in this repo.
- `eval.py`: Functions to evaluate the model's outputs against the ground truth on a validation subset.
- `grid_search.py`: What we used to optimize the hyperparameters of the model.
- `losses.py`: Definitions of loss functions used both in the training and evaluation processes.
- `train.py`: The training algorithm and associated helpers.
- `unet.py`: Our implementation of a U-Net.
- `watershed.py`: Helper functions to transform outputs produced by a 2 or 3-class U-Net into instance segmentations through the watershed algorithm.

## Running the notebook

The notebook `notebook.ipynb` can be run locally, but we recommend to run it on Google Colab,
which offers a free GPU (Just make sure to enable it under "Runtime -> Change Runtime Type")

No manual setup is needed. Simply run the notebook top to bottom to perform all the required setup steps such as downloading the dataset.
In the interest of time, however, we recommend using the pre-trained models over training your own. The notebook explains how to do so.

If you're looking at this README through GitHub (or other MarkDown renderer), you should see a Colab badge below.
Clicking it will take you directly to the notebook on Colab. (Colab might ask you for authorization to connect to your GitHub account).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CRefice/ml-segmentation-project/blob/master/notebook.ipynb)
