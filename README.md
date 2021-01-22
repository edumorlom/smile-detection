# Smile Detection

I created this smile/neutral CNN while I was bored during COVID-19 isolation to practice my ML skills.
NOTE: It can only detect neutral and smile faces, not intended to detect frowning, laughing, etc...

## Demo

![Demo](https://raw.githubusercontent.com/edumorlom/smile-detection/main/demo.gif)

## Install Dependencies

1. Install [Python](https://www.python.org/downloads/).
2. Run `python3 -m pip install -r requirements.txt` inside the `smile-detection/` directory.

## Train Model (Optional)

NOTE: By default, this GitHub repository comes with a pre-trained model.
Training a new model will take a while to run. Please be patient.

1. Open `smile-detection/Train.ipynb` with Jupyter Notebook.
2. In Jupyter Notebook, click: `Cell -> Run All`.
3. The model data will be written inside the `smile-detection/` directory as `smile.model/`.


## How to Run

1. Run `python3 main.py` inside the `smile-detection/` directory.

## Data

The data can be found inside `smile-detection/data.csv`, where each image is represented as an Numpy array for simplicty.
One can iterate through each image and read the numpy array as an image.
