# Car Purchase Amount Prediction

This notebook demonstrates a machine learning pipeline to predict the `Car Purchase Amount` based on various customer attributes.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
3.  [Data Preprocessing](#data-preprocessing)
4.  [Model Building and Training](#model-building-and-training)
5.  [Model Evaluation](#model-evaluation)
6.  [Prediction](#prediction)

## 1. Introduction
This project aims to predict the amount a customer is likely to spend on a car purchase. We use a dataset containing customer demographics and financial information to train a neural network model.

## 2. Data Loading and Initial Exploration
The first step involves loading the `Car_Purchasing_Data.csv` file into a pandas DataFrame. We then perform a quick check of the data's structure and contents.

### Key Steps:
-   Import necessary libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow.keras`).
-   Upload the CSV file using `files.upload()`.
-   Read the CSV into a DataFrame, handling potential `UnicodeDecodeError`.
-   Display the first and last few rows of the DataFrame to understand the data.
-   Visualize relationships between variables using `seaborn.pairplot()`.

## 3. Data Preprocessing
Before training the model, the data needs to be preprocessed. This involves:

### Key Steps:
-   Separating features (`X`) and target variable (`y`). The target variable is 'Car Purchase Amount'. Irrelevant columns like 'Customer Name', 'Customer e-mail', and 'Country' are dropped from the features.
-   Reshaping the target variable `y` for scaling.
-   Scaling both features (`X`) and the target variable (`y`) using `MinMaxScaler` to normalize the data between 0 and 1. This helps in faster and more stable model convergence.
-   Splitting the scaled data into training and testing sets using `train_test_split` with a 75:25 ratio.

## 4. Model Building and Training
A Artificial Neural Network (ANN) model is built using Keras (TensorFlow backend).

### Key Steps:
-   Initialize a `Sequential` model.
-   Add two `Dense` layers with 40 neurons each and 'relu' activation function. The first layer specifies `input_dim = 5` corresponding to the number of features.
-   Add an output `Dense` layer with 1 neuron and 'linear' activation for regression.
-   Compile the model using the 'adam' optimizer and 'mean_squared_error' as the loss function.
-   Train the model using the training data (`x_train`, `y_train`) for 100 epochs with a batch size of 50 and a validation split of 20%.

## 5. Model Evaluation
The training process includes tracking the loss on both the training and validation sets. This helps in understanding the model's performance and detecting overfitting.

### Key Steps:
-   Plot the training loss and validation loss over epochs to visualize the model's learning curve.

## 6. Prediction
After training, the model can be used to make predictions on new, unseen data.

### Key Steps:
-   Define a new sample data point for prediction.
-   Scale the new sample data using the same `MinMaxScaler` fitted on the training features (`x_scaler`).
-   Use the trained model (`model.predict()`) to predict the car purchase amount.
-   Inverse transform the predicted scaled value back to the original scale using `y_scaler.inverse_transform()` to get the actual predicted car purchase amount.
