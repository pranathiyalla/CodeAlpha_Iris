
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the existing trained model
model = joblib.load("iris_model.pkl")

# Flower species
species = ["Setosa", "Versicolor", "Virginica"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    flower = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(flower)
    result = species[prediction[0]]

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)
