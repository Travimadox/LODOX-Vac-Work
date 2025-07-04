# LODOX Start Up Guide
This is a small guide to help you get started with LODOX Training.

## Setup
1. Create a conda environment using the following command:
``
conda create --name Lodox python==3.10 -y
``

    This creates a conda environment with the name: `Lodox`

2. Activate the environment using the command:

    ``
    conda activate Lodox
    ``
3. Install the required packages using the command:

    ``
    pip install -r requirements.txt
    ``

## Running the Code
Some DICOM Images are in the folder `DICOM_Images`

The code uses command line parameters that you can set. The nost important parameters are: `--use_n2n`(to use n2n) `--use_n2v`(to use n2v) as shown below:

1. To use n2n:

    ``python Lodox.py --use_n2n True``

2. To use n2v:

    ``python Lodox.py --use_n2v True``

You can play around with the other arguments.

Good luck in handling the memory constraints. 
