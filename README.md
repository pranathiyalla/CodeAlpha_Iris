
# 🌸 Iris Flower Classification & Prediction Web App

## Project Overview
This project is a Machine Learning-based web application that predicts the species of an Iris flower based on its physical measurements.

The application uses a K-Nearest Neighbors (KNN) algorithm and provides an interactive website built using Flask, HTML and CSS.

## Objective
To classify Iris flowers into three species:
- Setosa
- Versicolor
- Virginica

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- HTML
- CSS
- Joblib

## Machine Learning Algorithm
K-Nearest Neighbors (KNN)

## Features
- Iris dataset analysis
- Data visualization
- Model training and testing
- Accuracy and confusion matrix evaluation
- Interactive web-based prediction
- Real-time flower species prediction

## Project Structure
- iris_classification.py - ML model implementation
- iris_model.pkl - Saved trained model
- app.py - Flask application
- templates/index.html - Website structure
- static/style.css - Website styling

## How to Run

1. Install the required libraries:
   pip install pandas numpy matplotlib seaborn scikit-learn flask joblib

2. Run the Flask application:
   python app.py

3. Open your browser and visit:
   http://127.0.0.1:5000

4. Enter the four flower measurements and click Predict Species.

## Output
The application predicts whether the flower belongs to Setosa, Versicolor or Virginica.

## Developed Using
Python, Machine Learning and Flask.
