# 🏆 Football Match Outcome Prediction
Machine Learning Models for Predicting Home Win, Away Win, or Draw

This project analyzes football match data and applies multiple machine learning models to predict match outcomes. The goal is to classify each match into one of three categories:

**- Home Win (H)
- Draw (D)
- Away Win (A)**

The workflow includes data cleaning, exploratory analysis in Power BI, model training, evaluation, and real-season predictions.

# 📊 Project Overview

The project is divided into several stages:

## 1. Data Collection & Cleaning

Football data covering multiple seasons.

Data cleaning and preprocessing performed in Power BI.

Removal of missing values, normalization, and encoding of categorical variables.

## 2. Machine Learning Models Used

Multiple models were trained and compared to determine the best-performing classifier:

Decision Tree

Support Vector Machine (SVM)

One-vs-All (OvA) Classifier

One-vs-One (OvO) Classifier

Logistic Regression

Random Forest

XGBoost

Each model was evaluated using:

Accuracy

Classification report

Confusion matrix

Feature importance (when applicable)

## 🧠 Why One-vs-One (OvO)?

Through experimentation, the OvO approach consistently achieved the best performance for this three-class classification problem.
Therefore, the final predictive scripts use the OvO strategy.

# 🔮 Final Prediction Scripts

This repository includes two final prediction scripts, each using the OvO method:

### 1️⃣ Predicting 2024/2025 Using Only the 2023/2024 Season

Trains the model exclusively on the 2023/2024 league data.

Predicts match outcomes for the 2024/2025 season.

Useful for analyzing how well a single-season model generalizes.

### 2️⃣ Predicting 2024/2025 Using Data from 2019–2024

Uses all matches from 2019/2020 to 2023/2024.

#Provides a richer dataset and better context for model learning.

Shows feature importance and a classification report for model evaluation.

Offers improved predictive power compared to the single-season approach.

# 📈 Insights Included

Comparative model performance

Feature importance analysis

Model evaluation metrics

Power BI dashboards for visualization

Season predictions for 2024/2025
