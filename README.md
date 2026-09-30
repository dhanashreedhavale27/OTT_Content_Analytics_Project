# OTT Content Analytics & Rating Predictor

## 1. Project Overview

This project uses the Kaggle "Netflix Movies and TV Shows" dataset to build an interactive Streamlit application.

The project has two major purposes:

1. Explore OTT content using data analytics.
2. Predict the rating category of a new OTT title using Machine Learning.

## 2. Kaggle Dataset

Dataset:
Netflix Movies and TV Shows

Kaggle:
https://www.kaggle.com/datasets/shivamb/netflix-shows

The dataset contains Netflix movie and TV show listings with fields such as type, title, director, cast, country, date added, release year, rating, duration and genres.

## 3. Project Structure

OTT_Content_Analytics_Project/
|
|-- app.py
|-- train.py
|-- predict.py
|-- download_dataset.py
|-- requirements.txt
|-- README.md
|-- REPORT_CONTENT.txt
|
|-- data/
|   |-- netflix_titles.csv
|
|-- models/
    |-- ott_rating_model.pkl
    |-- model_metrics.txt

## 4. Installation

Create/open the project folder and run:

pip install -r requirements.txt

## 5. Download Dataset

Run:

python download_dataset.py

This downloads the dataset from Kaggle using kagglehub.

## 6. Train Model

Run:

python train.py

The trained Random Forest model will be saved as:

models/ott_rating_model.pkl

## 7. Run Streamlit

Run:

streamlit run app.py

## 8. ML Method

The target variable is the OTT content rating category.

Input features:
- Content type
- Release year
- Duration
- Primary country
- Primary genre

Categorical values are converted using One-Hot Encoding. Missing numeric values are handled using median imputation.

The classifier used is Random Forest.

## 9. Enhancements over the Reference Project

- Multiple input features instead of one input.
- Categorical preprocessing.
- Dataset explorer.
- Interactive dashboard.
- Release trend chart.
- Rating distribution chart.
- Prediction probability display.
- Model evaluation metrics.
- Kaggle dataset download script.
- Separate prediction module.
