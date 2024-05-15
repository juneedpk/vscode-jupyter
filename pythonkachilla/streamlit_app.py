import streamlit as st
# adding title of app
st.title('Paddy Disease Prediction App')

# adding text to app

st.write('Through this application we can do paddy disease prediction')

# insert image

from PIL import Image

# user input in the form of image

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # prediction
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img(uploaded_file, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    from keras.models import load_model
    model = load_model('model.h5')
    result = model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'Brown Spot'
    elif result[0][1] == 1:
        prediction = 'Healthy'
    elif result[0][2] == 1:
        prediction = 'Hispa'
    elif result[0][3] == 1:
        prediction = 'Leaf Blast'
    else:
        prediction = 'Sheath Blight'
    st.write(prediction)


