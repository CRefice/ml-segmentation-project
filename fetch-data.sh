#!/bin/bash
set -e

if [[ -d dataset ]]; then
	echo "Dataset already downloaded."
	exit 0
fi
curl -o dataset.zip 'https://www.ebi.ac.uk/biostudies/files/S-BSST265/dataset.zip'
unzip dataset.zip -ddataset
rm dataset.zip
