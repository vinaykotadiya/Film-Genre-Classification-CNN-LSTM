Film Genre Classification Using CNN and LSTM
Overview
This project involves building and evaluating two deep learning models to classify film genres using multimodal data:

CNN (Convolutional Neural Network) to classify film posters (images) by genre.

LSTM (Long Short-Term Memory) network to classify film overviews (text) by genre.

The dataset contains film posters in JPEG format and film overview texts sourced from the Internet Movie Database (IMDb). The goal is to develop efficient GPU-accelerated data processing pipelines using TensorFlow and Keras, then critically evaluate the models' performance.

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
