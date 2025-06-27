This project involves building and evaluating two deep learning models to classify film genres using multimodal data:

CNN (Convolutional Neural Network) to classify film posters (images) by genre.

LSTM (Long Short-Term Memory) network to classify film overviews (text) by genre.

The dataset contains film posters in JPEG format and film overview texts sourced from the Internet Movie Database (IMDb). The goalCertainly! Here's a clean, well-formatted README file for your project—completely without images and easy to read:

---

# Film Genre Classification Using CNN and LSTM

## Project Overview

This project involves developing two deep learning models to classify film genres based on multimodal data:

* **Convolutional Neural Network (CNN)** classifies film posters (images) by genre.
* **Long Short-Term Memory (LSTM)** network classifies film overviews (text) by genre.

Using a dataset of film posters and overviews from IMDb, the models are trained and evaluated leveraging TensorFlow and Keras with GPU acceleration. The project emphasizes efficient data processing pipelines and critical evaluation of model performance.

## Learning Objectives

* Understand how GPUs accelerate data processing.
* Develop GPU-accelerated data pipelines using TensorFlow’s `tf.data` API.
* Build and train CNN and LSTM models for image and text classification.
* Critically evaluate model results with insights into performance, limitations, and improvements.

## Dataset

* **Multimodal\_IMDB\_dataset.zip**: Contains JPEG film posters and text-based film overviews, along with genre labels.

## Repository Contents

* `Keras_Assignment_Dec2024.ipynb` — Main notebook with:

  * Data preprocessing (image & text) using TensorFlow pipelines.
  * CNN and LSTM model definitions and training routines.
  * Model evaluation and prediction on sample films.

## Methodology

1. **Data Processing**

   * Implement image processing functions and text encoding.
   * Build optimized TensorFlow data pipelines using `tf.data`.

2. **Model Architecture**

   * Construct and compile a CNN for poster classification based on given model specifications.
   * Build an LSTM model with embedding layers for overview classification.

3. **Training**

   * Use callbacks for monitoring and train models on GPU.

4. **Evaluation**

   * Predict genres for selected films, analyze results, and critically discuss findings in an accompanying report.

## Usage Instructions

* Run the notebook in a GPU-enabled environment such as Google Colab.
* Install TensorFlow 2.x and required dependencies.
* Load the dataset and follow the notebook steps for preprocessing, training, and evaluation.
* Review predictions and performance metrics for insights.

## Reporting

A 2-3 page report accompanies the code, providing a detailed critical analysis of model accuracy, challenges encountered, and suggestions for future work.
