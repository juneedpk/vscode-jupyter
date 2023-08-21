import numpy as np
import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
from ydata_profiling import ProfileReport


#web app title
st.markdown(''' 
 # **Exploratory data analysis Web App**
This is the **EDA App** created in Streamlit using the **pandas-profiling** library.''')


# how to upload file

with st.sidebar.header('1. Upload your CSV data (.csv)'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


if uploaded_file is not None:
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('press to use example data'):
    # example data

     def load_csv():
        csv = pd.read_csv('Rainfall_PAK.csv')
        return csv
     df = load_csv()
     pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)