Fake Account Detection Model
Overview
This project implements a Random Forest Classifier to detect fake accounts using a structured dataset. The model is trained on labeled data and can predict whether a given account is real (0) or fake (1).

Features
Binary classification (Real vs Fake)

Random Forest algorithm with 100 estimators

Model evaluation metrics: Accuracy, Precision, F1 Score, Confusion Matrix

Model persistence using joblib

Batch processing: saves each dataset row as an individual CSV file for testing

Requirements
Python 3.x

pandas

numpy

matplotlib

scikit-learn

joblib

Dataset
The dataset (Dataset.csv) should contain:

Feature columns (all columns except fake)

Target column named fake (0 = Real, 1 = Fake)

Model Performance
Based on the evaluation metrics:

Accuracy: ~91.38%

Precision: ~91.38%

F1 Score: ~91.38%

The model shows balanced performance across both classes.

Project Structure

├── Dataset.csv                 # Input dataset
├── fake_account_model.pkl      # Saved trained model
├── rows_output/                # Folder containing individual row CSV files
│   ├── row1.csv
│   ├── row2.csv
│   └── ...
└── Untitled.ipynb              # Main Jupyter notebook

Usage
1. Train the Model
Run the notebook cells to:

Load and split the dataset (80% train, 20% test)

Train the Random Forest classifier

Evaluate performance

Save the model as fake_account_model.pkl

2. Predict on a Single Row

import pandas as pd
import joblib

# Load model
model = joblib.load("fake_account_model.pkl")

# Load row data
df = pd.read_csv("rows_output/row512.csv")

# Predict
prediction = model.predict(df)
print("Fake" if prediction[0] == 1 else "Real")

3. Export All Rows
The notebook includes code to export each row from the dataset as an individual CSV file in the rows_output/ folder for batch testing.

Evaluation Metrics
Metric	Score
Accuracy	0.9138
Precision	0.9138
F1 Score	0.9138


Classification Report

  precision    recall  f1-score   support
0       0.91      0.91      0.91        58
1       0.91      0.91      0.91        58

Confusion Matrix
The confusion matrix visualization shows the distribution of true positives, true negatives, false positives, and false negatives.

Output
Model file: fake_account_model.pkl

Individual row CSVs: rows_output/row*.csv

Notes
The model uses random_state=42 for reproducibility

Stratified split ensures balanced class representation in train/test sets

