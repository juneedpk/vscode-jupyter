import streamlit as st
import pandas as pd
import joblib

# title
st.header('cat vs dog App')

# input  left bar

height = st.number_input('height', min_value=0.0, max_value=200.0, step=1)


weight = st.number_input('weight', min_value=0.0, max_value=200.0, step=1)

# dropdown list

eyes = st.selectbox('select eyes color', ['blue', 'brown'])

# radio button

if st.button('submit'):

    model = joblib.load('st.pkl')
    data = [[height, weight, eyes]]
    X = pd.DataFrame(data, columns=['height', 'weight', 'eyes'])
    X = X.replace({'eyes': {'blue': 0, 'brown': 1}})


    # predict
pred = model.predict(X)[0]

st.text(pred)
