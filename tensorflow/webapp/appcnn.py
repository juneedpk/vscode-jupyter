import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os



# Load the trained model


model = tf.keras.models.load_model('vscode-jupyter-1\model.h5') 

# Streamlit app
st.title("MNIST Digit Classification")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a digit image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for the model prediction
    img_array = np.array(image.resize((28, 28))).mean(axis=-1, keepdims=True)
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write("")
    st.write("Class Predicted:", predicted_class)
    st.write("Confidence:", prediction[0][predicted_class])

# Additional content (e.g., model summary, training history)
import matplotlib.pyplot as plt

if st.checkbox("Show Model Summary"):
    st.subheader("Model Summary")
    st.text(model.summary())

if st.checkbox("Show Training History"):
    st.subheader("Training History")
    fig, ax = plt.subplots()
    ax.plot(model.history['accuracy'], label='Training Accuracy')
    ax.plot(model.history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

