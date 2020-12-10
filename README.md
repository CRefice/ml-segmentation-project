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

# GitHub repo as requirement for project

- Create a `requirements.txt` containing all the dependencies for your project (tip: use `pip freeze` before and after)
- In the `requirements.txt` enter this line if you want to add a GitHub repo as a dependency (for SSH)
>	git+git://github.com/path/to/package

then

> 	pip install -r requirements.txt

to 'freeze' the requirements for your project:

> 	pip freeze > requirements.txt

# Todo after meeting 19/11
- [x] Add Sigmoid / ReLU after last convolution to fix the strange Dice loss behavior
- [x] Train with bigger images / smaller batch size / more epochs
- [x] Adapt to 3-class classifier to include the boundary class
- [x] Do the Train/Test split correctly
- [x] Experiment with combining Dice with PCE loss
- [x] Use Adam learning rate for SGD, or perform a GridSearch
- [x] Use Min/Max normilization in the Transform Lambda

# Todo after meeting 03/12
- [ ] Work out the report chapters
- [ ] Start writing
- [X] Create own implementation of Dice loss which works with our tensors
- [X] Fidle around with skikit.find_boundaries (not exact method name)
- [X] Modify predict in unet.py to not use thresholding for multiclass case
- [X] Add weights for the multi-class cross-entropy
- [X] Figure out how Watershed works and apply it to divide the image in different cells -> to get cell_id's
- [ ] StarDist
  - [ ] Running SD on our dataset
  - [ ] Find out which metric StarDist uses (IoU)
  - [ ] Compare SD and UNet with multiple classes

Useful links:
- https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html 
- https://github.com/mpicbg-csbd/stardist/blob/master/extras/stardist_example_2D_colab.ipynb 
