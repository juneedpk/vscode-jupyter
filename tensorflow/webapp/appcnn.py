{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "\n",
    "# Load the trained model\n",
    "model = tf.keras.models.load_model('path_to_your_model')  # Replace 'path_to_your_model' with the actual path\n",
    "\n",
    "# Streamlit app\n",
    "st.title(\"MNIST Digit Classification\")\n",
    "\n",
    "# Upload image through Streamlit\n",
    "uploaded_file = st.file_uploader(\"Choose a digit image...\", type=\"jpg\")\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    # Display the uploaded image\n",
    "    image = Image.open(uploaded_file)\n",
    "    st.image(image, caption='Uploaded Image.', use_column_width=True)\n",
    "\n",
    "    # Preprocess the image for the model prediction\n",
    "    img_array = np.array(image.resize((28, 28))).mean(axis=-1, keepdims=True)\n",
    "    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0\n",
    "\n",
    "    # Make a prediction\n",
    "    prediction = model.predict(img_array)\n",
    "    predicted_class = np.argmax(prediction)\n",
    "\n",
    "    st.write(\"\")\n",
    "    st.write(\"Class Predicted:\", predicted_class)\n",
    "    st.write(\"Confidence:\", prediction[0][predicted_class])\n",
    "\n",
    "# Additional content (e.g., model summary, training history)\n",
    "if st.checkbox(\"Show Model Summary\"):\n",
    "    st.subheader(\"Model Summary\")\n",
    "    st.text(model.summary())\n",
    "\n",
    "if st.checkbox(\"Show Training History\"):\n",
    "    st.subheader(\"Training History\")\n",
    "    st.line_chart(history.history['accuracy'])\n",
    "    st.line_chart(history.history['val_accuracy'])\n",
    "    st.line_chart(history.history['loss'])\n",
    "    st.line_chart(history.history['val_loss'])\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
