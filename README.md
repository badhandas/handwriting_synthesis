# Handwriting Synthesis using Graves approach

This repository contains my attempt at implementing Alex Graves paper on sequence generation in Pytorch lightning module.
Please find the paper: https://arxiv.org/abs/1308.0850v5


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

A few examples:

![image](https://user-images.githubusercontent.com/87189221/125092442-a66ed480-e0d1-11eb-9518-2f0b008bd6e3.png)
Some results are not 100% but still better than season 8 ;-)
![image](https://user-images.githubusercontent.com/87189221/125092598-c900ed80-e0d1-11eb-830d-136846fa707a.png)
Everyone does :-/
![image](https://user-images.githubusercontent.com/87189221/125092642-d28a5580-e0d1-11eb-887f-44b1ee723b5e.png)

