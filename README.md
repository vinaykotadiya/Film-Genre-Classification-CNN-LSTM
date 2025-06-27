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

---

If you'd like, I can also help draft the critical report summary or CV bullet points for this project!
 is to develop efficient GPU-accelerated data processing pipelines using TensorFlow and Keras, then critically evaluate the models' performance.

Learning Outcomes
Demonstrate understanding of GPU acceleration in data processing.

Develop optimized GPU-accelerated data pipelines using TensorFlow tf.data API.

Implement and train CNN and LSTM models using TensorFlow and Keras.

Evaluate model performance critically with insights into strengths and weaknesses.

Dataset
Multimodal_IMDB_dataset.zip — contains film posters and film overview text files along with genre labels.

Project Structure
Keras_Assignment_Dec2024.ipynb — Main notebook implementing:

Data preprocessing pipelines for images and text.

Model definitions for CNN (image classification) and LSTM (text classification).

Model training with callbacks and GPU acceleration.

Evaluation and visualization of results with predictions on sample films.

Key Tasks
Data Processing

Build efficient TensorFlow pipelines for image and text data using the tf.data API.

Prepare vocabulary and encode text inputs with encoder.adapt().

Model Construction

Design and compile CNN based on provided model summary for poster classification.

Implement LSTM-based sequential model for overview classification using embedding layers.

Training

Define training callbacks and train models on GPU for specified epochs.

Evaluation and Reporting

Predict genres for selected films.

Plot posters and print overviews with predicted genres.

Write a critical report analyzing model performance, discussing successes, failures, and potential improvements.

Usage
Run the notebook in Google Colab or a local environment with GPU support.

Ensure TensorFlow and Keras are installed (TensorFlow 2.x recommended).

Load and preprocess the dataset as demonstrated in the notebook.

Train models and observe training logs and performance metrics.

Generate evaluation plots and predictions to support analysis.

Results and Analysis
The accompanying report (2-3 pages) critically evaluates the classification accuracy, discusses overfitting/underfitting issues, and compares the CNN and LSTM performance. Insights into data characteristics influencing results are also included.
