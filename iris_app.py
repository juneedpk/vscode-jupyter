import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Train the Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Create a Streamlit web application
st.title("Iris Species Prediction")

# User input for feature values
st.sidebar.header("Feature Values")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()))

# Predict the species using Logistic Regression
logistic_prediction = logistic_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

# Predict the species using Random Forest Classifier
rf_prediction = rf_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

# Display the predicted species for both models
st.write("## Predicted Species")
st.write(f"Logistic Regression Model: {iris.target_names[logistic_prediction]}")
st.write(f"Random Forest Classifier Model: {iris.target_names[rf_prediction]}")

# Display the model accuracies
st.write("## Model Accuracies")
logistic_accuracy = accuracy_score(y_test, logistic_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
st.write(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")
st.write(f"Random Forest Classifier Accuracy: {rf_accuracy:.2f}")

# Display classification reports
st.write("## Classification Reports")
st.write("Logistic Regression Classification Report:")
st.code(classification_report(y_test, logistic_model.predict(X_test), target_names=iris.target_names))
st.write("Random Forest Classifier Classification Report:")
st.code(classification_report(y_test, rf_model.predict(X_test), target_names=iris.target_names))
