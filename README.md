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

## Project dependencies

As previously mentioned, running the notebook on Colab should take care of installing the required dependencies, but if running it locally, you'll need to install (at least some of) them yourself.
Here's a list of required dependencies, along with the latest version number they have been tested with:

- `csbdeep` 0.6.1
- `gdown` 3.6.4
- `imagecodecs` 2020.5.30
- `ipykernel` 4.10.1
- `ipython` 5.5.0
- `ipywidgets` 7.5.1
- `Keras` 2.3.1
- `Keras-Applications` 1.0.8
- `Keras-Preprocessing` 1.1.2
- `matplotlib` 3.2.2
- `notebook` 5.3.1
- `numpy` 1.18.5
- `scikit-image` 0.16.2
- `scikit-learn` 0.22.2.post1
- `scipy` 1.4.1
- `sklearn` 0.0
- `stardist` 0.6.1
- `tifffile` 2020.9.3
- `torch` 1.7.0+cu101
- `torchsummary` 1.5.1
- `torchvision` 0.8.1+cu101
- `tqdm` 4.41.1
