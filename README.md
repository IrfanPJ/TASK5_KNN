 K-Nearest Neighbors on Iris Dataset

Overview

This script classifies the Iris dataset using the K-Nearest Neighbors (KNN) algorithm. It performs data preprocessing, trains the model with different values of k, evaluates performance, and visualizes the results.

Steps

Load Data: The Iris dataset is loaded and prepared.

Preprocessing: Features are scaled for model training.

Model Training: KNN is applied with k values from 1 to 5, and accuracy is calculated.

Evaluation: A confusion matrix is generated for the best model (k=3).

Visualization: A scatter plot of the first two features is shown, color-coded by species.

Output

Accuracy for different values of k.

Confusion matrix for k=3.

A scatter plot of the dataset.

Requirements

pandas, sklearn, matplotlib libraries.