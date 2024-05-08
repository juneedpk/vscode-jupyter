import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score



# Function to preprocess data
def preprocess_data(data):
    # Fill missing values using IterativeImputer
    imputer = IterativeImputer()
    data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # One-hot encode categorical variables
    
    data_encoded = pd.get_dummies(data_filled, drop_first=True)

    # Scale the data if features are not in scale
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_encoded)

    return data_scaled, data_encoded,data_filled, data_encoded.columns

# Function to train and evaluate model
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, r2

# Main function
def main():
    st.title("Machine Learning Application")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload Dataset (CSV, XLSX, etc.)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Basic data information
        st.write("Basic Data Information:")
        st.write(data.head())
        st.write("Data Types:")
        st.write(data.dtypes)

        # Select features and targets
        features = st.multiselect("Select Columns as Features", data.columns)
        targets = st.multiselect("Select Columns as Targets", data.columns)

        # Preprocess data if the data is cateogrical or object

        data_scaled, data_encoded,data_filled, data_encoded.columns = 
    
        # Select model
        model_selection = st.sidebar.selectbox("Select Model", ["Linear Regression", "Decision Tree", "Support Vector Machine"])

        if model_selection == "Linear Regression":
            model = LinearRegression()
        elif model_selection == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model_selection == "Support Vector Machine":
            model = SVR()

        # Train-test split
        X = processed_data[:, [data.columns.get_loc(col) for col in columns if col in features]]
        y = processed_data[:, [data.columns.get_loc(col) for col in columns if col in targets]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and evaluate model
        y_pred, mse, r2 = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)

        # Inverse transform data
        data_inverse_imputed = inverse_transform_data(X_test, imputer, scaler, columns)

        # Display predictions
        st.write("Predictions:")
        st.write(pd.DataFrame(data_inverse_imputed, columns=columns))
        st.write("Mean Squared Error:", mse)
        st.write("R^2 Score:", r2)

if __name__ == "__main__":
    main()
