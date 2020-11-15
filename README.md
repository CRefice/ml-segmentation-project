# Data acquisition

The data isn't stored in this repository to save space. [You can download the data through this link.](https://www.ebi.ac.uk/biostudies/files/S-BSST265/dataset.zip) or through the `fetch-data.sh` script included in the repo.

# Running the notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CRefice/ml-segmentation-project/blob/master/notebook.ipynb)

# Roadmap

Goal: Segment biomedical images of cells under a microscope into the seperate cells.
Data:
- Bunch of TIFFS, Grayscale
- - Microscope image itself
- - Labelled with grayscale value, black is background, every other value is a label

Steps:
1. Read the input data to work with it
- Transform this data to a form which can be input to the UNet
2. Train the UNet
- First, to recognize background/foreground
- Second, to identify each seperate cell
