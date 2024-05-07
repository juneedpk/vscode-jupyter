import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# lets create title

st.title("Data Analysis with Python")
st.subheader("Simple Data Analysis creating using Streamlit")

# create drop down menu for multiple dataset
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "House Rent", "mars dataset"))

# load dataset

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = pd.read_csv("iris.csv")
    elif dataset_name == "House Rent":
        data = pd.read_csv("House_Rent.csv")
    else:
        data = pd.read_csv("mars_dataset.csv")
    return data

# button to upload custom dataset

if st.sidebar.button("Upload Custom Dataset"):
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv","xlsx"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        st.write("Please upload a file")

# displaying dataset

data = get_dataset(dataset_name)
st.write(data)








