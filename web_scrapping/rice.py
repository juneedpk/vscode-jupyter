
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

# Load the pre-trained model (assuming it's saved as 'model.h5')
model = load_model('rice.h5')

# Define class labels (modify these based on your actual classes)
class_names = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 
               'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

def predict_disease(image_file):
  """
  Predicts the disease from a uploaded image.
  Args:
      image_file: Uploaded image file object.
  Returns:
      A tuple containing the predicted class index and class name.
  """
  img = image.load_img(image_file, target_size=(224, 224))
  img_data = image.img_to_array(img)
  img_data = img_data / 255.0
  img_data = img_data.reshape(1, 224, 224, 3)
  prediction = model.predict(img_data)
  predicted_class_index = np.argmax(prediction)
  predicted_class_name = class_names[predicted_class_index]
  return predicted_class_index, predicted_class_name

st.title("Paddy Disease Classification")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
  # Display uploaded image
  st.image(uploaded_file, width=250)
  
  # Make prediction
  predicted_class_index, predicted_class_name = predict_disease(uploaded_file)

  # Display prediction results
  st.success(f"Predicted Disease: {predicted_class_name}")
  


