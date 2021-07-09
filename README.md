# Handwriting Synthesis using Graves approach

This repository contains my attempt at implementing Alex Graves paper on sequence generation in Pytorch lightning module.

Steps:

Data pre-processing: Run the following code
_python data_preprocessing.py_

Training: Run the following code
_python lit.py_

Checking: Run the following code
_python check.py_

Data needs to be stored in data directory
Pretrained model is stored in trained_model directory.
Output results are stored in generated image directory

Details:
Change the directory paths according to your paths.
Uncomment the line num 220 to 227 in lit.py for training and comment again for checking.
Give the path of the saved pretrained model in generate.py
Then give the text input in check.py file and run check.py to get the result.