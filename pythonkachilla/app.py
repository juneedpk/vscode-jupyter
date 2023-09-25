import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


# Load the pre-trained Logistic Regression model
#model = joblib.load('javaid.pkl')



df = pd.read_csv('https://raw.githubusercontent.com/gerchristko/Streamlit-ML-App/main/data.csv')
df = df.dropna()

df['Eye'] = df['Eye'].astype('category').cat.codes

# Create a list of features

X = df[['Height', 'Weight','Eye']]
y = df['Species']



model = LogisticRegression().fit(X, y)



# Define a function to predict the species
def predict_species(height, weight, eye):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({'Height': [height], 'Weight': [weight], 'Eye': [eye]})
    

    # Make a prediction
    prediction = model.predict(input_data)
    
    # Return the predicted species
    return prediction[0]

# Streamlit UI
st.title("Species Prediction App")
st.sidebar.header("Input Parameters")

# User input for height, weight, and eye color
height = st.sidebar.slider("Height (cm)", 0.0, 300.0, 150.0)
weight = st.sidebar.slider("Weight (kg)", 0.0, 200.0, 70.0)
eye_color = st.sidebar.selectbox("Eye Color", df['Eye'])

# Make a prediction and display the result
if st.sidebar.button("Predict"):
    species = predict_species(height, weight, eye_color)
    st.write(f"Predicted Species: {species}")

# Optionally, you can add some explanation or information about the model
st.write("""
### Model Information
This app uses a pre-trained Logistic Regression model to predict the species based on height, weight, and eye color.
""")

# Optionally, you can add some information about the dataset or a link to the source.
st.write("""
### Dataset Information
The dataset used for training this model can be found [here](https://raw.githubusercontent.com/gerchristko/Streamlit-ML-App/main/data.csv).
""")

# Optionally, you can add some information about the app and its purpose.
st.write("""
### About
This Streamlit app is for educational purposes and demonstrates how to deploy a machine learning model using Streamlit.
""")