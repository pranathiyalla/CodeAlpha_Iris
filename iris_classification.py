import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load Iris dataset
iris = load_iris()

# Convert dataset into DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Display first five rows
print(df.head())

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nFlower Species:")
print(iris.target_names)
# Add target species to the DataFrame
df["species"] = iris.target

# Convert numerical labels into flower names
df["species"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# Display species count
print("\nSpecies Count:")
print(df["species"].value_counts())

# Create a scatter plot
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species"
)

plt.title("Iris Flower Classification")
# Separate input and output data
X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)
# Save the trained model
joblib.dump(model, "iris_model.pkl")

print("Model saved successfully!")

print("\nModel Training Completed!")

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Generate classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))
# Plot confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Confusion Matrix")
plt.show()
# Predict flower species using user input

print("\nEnter Flower Measurements")

sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

# Store user input
new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

# Predict flower species
prediction = model.predict(new_flower)

# Display predicted species
predicted_species = iris.target_names[prediction[0]]

print("\nPredicted Flower Species:", predicted_species)