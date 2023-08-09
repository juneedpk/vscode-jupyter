import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score



# make containers

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


with header:
    st.title('Welcome to my awesome data science project')
    st.text('In this project I look into the titanic dataset')

with dataset:
    st.header('Titanic Data')
    st.text('I found this dataset on kaggle')

    #import data
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head(10))
    st.subheader('Passenger Sex distribution')
    st.bar_chart(df['sex'].value_counts())

    st.subheader('Passenger Age distribution')
    st.bar_chart((df['age'].value_counts()).head(10))


##with features:
    #st.header('The features I created')
    #st.text('I created these features')


with model_training:    
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes')

    # making columns

    input,display = st.columns(2)
    #first column selection points
    max_depth = input.slider('how many people you know?',min_value=5,max_value=100,step=5,value=20)

    n_estimators = input.selectbox('How many trees do you want in your forest?',options=[50,100,200,300,'No limit'])

    #input features from users
    input_features = input.text_input('enter feature here','Type here')
    input.write(df.columns)

    #define x and y

    x = df[[input_features]]
    y = df[['fare']]

    #meachine learning model creation and fitting

    model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators).fit(x,y)

    

    prediction = model.predict(y)

    #display metrices

    display.subheader('Mean absolute error of the model is:')
    display.write(mean_absolute_error(y,prediction))
    display.subheader('Mean squared error of the model is:')
    display.write(mean_squared_error(y,prediction))
    display.subheader(' R squared score of the model is:')
    display.write(r2_score(y,prediction))


