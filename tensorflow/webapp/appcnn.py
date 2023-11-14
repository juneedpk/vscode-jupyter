import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('vscode-jupyter-1\tensorflow\mnist_cnn.h5')  # Replace 'path_to_your_model' with the actual path

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
if st.checkbox("Show Model Summary"):
    st.subheader("Model Summary")
    st.text(model.summary())

if st.checkbox("Show Training History"):
    st.subheader("Training History")
    st.line_chart(history.history['accuracy'])
    st.line_chart(history.history['val_accuracy'])
    st.line_chart(history.history['loss'])
    st.line_chart(history.history['val_loss'])
