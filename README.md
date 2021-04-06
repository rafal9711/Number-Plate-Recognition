
## Purpose of the project:
Project was made for Computer Vision classes.
The aim of the project was to write a license plate recognition program.

## Project assumptions:
* number plates in the pictures will be inclined not more than ± 45 degrees to the horizontal position,
* the longer edge of the number plate covers over one third of the photo width,
* the angle between the optical axis of the camera and the plate plane does not exceed 45 degrees,
* ordinary plates are photographed, with black characters on a white background (7 characters).

## Requirements:
* Programs should be written in Python version 3.7, using the OpenCV library.
* It is possible to use external libraries (e.g. scikit-image), but it is not allowed to use external OCR modules or ready-made, trained models that enable reading characters.

## Setup:
```
$ pip install -r requirements.txt
```
To run this project, you need to pass two arguments:
path to images directory,
path to output json file
```
$ python main.py images_dir results_file_json
```
