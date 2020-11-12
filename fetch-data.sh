#!/bin/bash
set -e

curl -o dataset.zip 'https://www.ebi.ac.uk/biostudies/files/S-BSST265/dataset.zip'
unzip dataset.zip -ddataset
rm dataset.zip
