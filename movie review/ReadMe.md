# Movie Sentiment Analysis Application

## Overview

This project demonstrates a movie sentiment analysis application using a Long Short-Term Memory (LSTM) model for textual data. The model is built using TensorFlow and Keras, trained on the IMDB dataset to classify movie reviews as positive or negative. The application is deployed using Gradio for an interactive user interface.
You can run this one google colab make sure to run :
1. sentimentpro_modelGenCode.ipynb
2. ModelandUI.ipynb
## Note : dont put any other file mannually in colab except IMDB Dataset.csv
3. after running   'sentimentpro_modelGenCode.ipynb' successfully youll get "model.h5" and "tokenizer.pkl" generated which you have to download and upload in "ModelandUI.ipynb"

## Project Structure

- **IMDB Dataset**: The dataset used for training and testing the model.
- **Model Building**: The LSTM model is built and trained.
- **Deployment**: The model is deployed using Gradio for interactive usage.
### Prerequisites

- Python 3.x
- TensorFlow and Keras
- Pandas and NumPy
- Gradio for deployment

### Installation

To set up the environment, run:

`pip install tensorflow pandas numpy gradio`