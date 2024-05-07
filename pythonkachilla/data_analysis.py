import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# set option to avoid warning
st.set_option('deprecation.showPyplotGlobalUse', False)


# lets create title

st.title("Data Analysis with Python")
st.subheader("Simple Data Analysis creating using Streamlit")

# create drop down menu for multiple dataset
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Flights", "Taxis"))

# load dataset

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = sns.load_dataset("iris")
    elif dataset_name == "Flights":
        data = sns.load_dataset("flights")
    else:
        data = sns.load_dataset("taxis")
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

# displaying data columns and rows

if st.sidebar.checkbox("Discriptive Summary"):
    st.write(data.describe())


if st.sidebar.checkbox("Data Information"):
    st.write(data.dtypes)
   
st.write('Total rows:',data.shape[0])
   
st.write('Total columns:',data.shape[1])

# create a function to print null values greater than 0

def missing_values(data):
    missing = data.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    return missing

if st.sidebar.checkbox("Missing Values"):
    st.write(missing_values(data))

# create a function to select custom columns and plot that columns

def plot_data(data):
    columns = data.columns
    selected_columns = st.multiselect("Select Columns", columns)
    if selected_columns:
        plot_type = st.selectbox("Select Plot Type", ["area", "bar", "line", "hist", "box", "kde",'scatter'])
        if plot_type == "area":
            st.write(data[selected_columns].plot(kind=plot_type))
            st.pyplot()
        elif plot_type == "bar":
            st.write(data[selected_columns].plot(kind=plot_type))
            st.pyplot()
        elif plot_type == "line":
            st.write(data[selected_columns].plot(kind=plot_type))
            st.pyplot()
        elif plot_type == "hist":
            st.write(data[selected_columns].plot(kind=plot_type))
            st.pyplot()
        elif plot_type == "box":
            st.write(data[selected_columns].plot(kind=plot_type))
        elif plot_type == "scatter":
            st.write(px.scatter(data,x=selected_columns[0],y=selected_columns[1]))
            st.pyplot()
        else:
            st.write(data[selected_columns].plot(kind=plot_type))
            st.pyplot()

if st.sidebar.checkbox("Plot Data"):
    plot_data(data)

# create pairplot 

if st.sidebar.checkbox("Pairplot"):
    st.write(sns.pairplot(data))
    st.pyplot()

# select the columns which are numeric and create correltion plot

def numeric_columns(data):
    return data.select_dtypes(include=[np.number]).columns

if st.sidebar.checkbox("Correlation Plot"):
    st.subheader("Correlation Plot")
    numeric = numeric_columns(data)
    selected_columns = st.multiselect("Select Columns", numeric)
    if selected_columns:
        st.write(data[selected_columns].corr())
        st.write(sns.heatmap(data[selected_columns].corr(), annot=True))
        st.pyplot()

























