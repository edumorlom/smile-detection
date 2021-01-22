# Smile Detection

I created this smile/neutral CNN while I was bored during COVID-19 isolation to practice my ML skills.
NOTE: It can only detect neutral and smile faces, not intended to detect frowning, laughing, etc...

## Demo

![Demo](https://raw.githubusercontent.com/edumorlom/smile-detection/main/demo.gif)

## How to Run

1. Install [Python](https://www.python.org/downloads/).
2. Install Python dependencies inside `smile-detection` directory: `python3 -m pip install -r requirements.txt`.
2. Run `python3 main.py` inside the `smile-detection/` directory.

## Data

The data can be found inside `smile-detection/data.csv`, where each image is represented as an Numpy array for simplicty.
One can iterate through each image and read the numpy array as an image.
