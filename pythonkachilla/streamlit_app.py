import streamlit as st
# adding title of app
st.title('Paddy Disease Prediction App')
# adding image to app
from PIL import Image
img = Image.open("E:\vscode\vscode-jupyter-1\paddy.PNG")
# adding text to app
st.write('Through this application we can do paddy disease prediction')


