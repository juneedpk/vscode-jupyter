import streamlit as st
import seaborn as sns



st.header("This is a dataset of IRIS flowers")
st.subheader("This is explained in the datset description")



df = sns.load_dataset('iris')
st.write(df.describe())
st.bar_chart(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]) # bar chart
st.line_chart(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]) # line chart
st.area_chart(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]) # area chart
