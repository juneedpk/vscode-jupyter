import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, span):
    return data.ewm(span=span, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def calculate_obv(data):
    obv = []
    prev_obv = 0
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(prev_obv + data['Volume'].iloc[i])
            prev_obv += data['Volume'].iloc[i]
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(prev_obv - data['Volume'].iloc[i])
            prev_obv -= data['Volume'].iloc[i]
        else:
            obv.append(prev_obv)
    return pd.Series(obv, index=data.index[1:])

def calculate_stochastic_oscillator(data, window=14):
    lowest_low = data['Low'].rolling(window=window).min()
    highest_high = data['High'].rolling(window=window).max()
    data['%K'] = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    data['%D'] = data['%K'].rolling(window=3).mean()
    data['10_SMA'] = calculate_sma(data['Close'], window=10)
    data['20_SMA'] = calculate_sma(data['Close'], window=20)
    data['12_EMA'] = calculate_ema(data['Close'], span=12)
    data['26_EMA'] = calculate_ema(data['Close'], span=26)
    data['RSI'] = calculate_rsi(data['Close'])
    data['UpperBB'], data['LowerBB'] = calculate_bollinger_bands(data['Close'])
    return data

def calculate_adx(data, window=14):
    delta_high = data['High'].diff()
    delta_low = -data['Low'].diff()
    true_range = pd.concat([delta_high, delta_low], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    up_move = (data['High'] - data['High'].shift(1)).apply(lambda x: x if x > 0 else 0)
    down_move = (data['Low'].shift(1) - data['Low']).apply(lambda x: x if x > 0 else 0)
    pos_dm = up_move.rolling(window=window).mean()
    neg_dm = down_move.rolling(window=window).mean()
    pos_di = (pos_dm / atr) * 100
    neg_di = (neg_dm / atr) * 100
    dx = ((pos_di - neg_di).abs() / (pos_di + neg_di).abs()) * 100
    adx = dx.rolling(window=window).mean()
    return adx

def create_and_train_model():
    # Fetch data from yfinance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)  
    data = yf.download('AAPL', start=start_date, end=end_date)

    # Calculate additional indicators
    data = calculate_stochastic_oscillator(data)
    data['OBV'] = calculate_obv(data)
    data['ADX'] = calculate_adx(data)

    # Prepare the data
    data = data.dropna()  

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Adj Close', 'OBV', '10_SMA', '20_SMA', '12_EMA', '26_EMA', 'RSI', 'UpperBB', 'LowerBB', '%K', '%D', 'ADX']])

    # Generate training data
    X_train = []
    y_train = []

    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i])
        y_train.append(scaled_data[i, 0])  # Only the closing price is the target

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 12))  # 12 features now

    # Define the LSTM model with ReLU activation and Dropout layer
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=2)

    # Save the model and scaler
    model.save('stock_price_model.h5')
    np.save('scaler.npy', scaler)
    
    return model, scaler

def predict_stock_price(input_date, company_ticker):
    # Load the model and scaler
    model = tf.keras.models.load_model('stock_price_model.h5')
    scaler = np.load('scaler.npy', allow_pickle=True).item()

    # Check if the input date is a valid date format
    try:
        input_date = pd.to_datetime(input_date)
    except ValueError:
        st.error("Invalid Date Format. Please enter date in YYYY-MM-DD format.")
        return

    # Fetch data from yfinance
    end_date = input_date
    start_date = input_date - timedelta(days=120)  
    data = yf.download(company_ticker, start=start_date, end=end_date)

    # Calculate additional indicators
    data = calculate_stochastic_oscillator(data)
    data['OBV'] = calculate_obv(data)
    data['ADX'] = calculate_adx(data)
    if len(data) < 60:
        st.warning("Not enough historical data to make a prediction. Try an earlier date.")
        return

    # Scale the data
    scaled_data = scaler.transform(data[['Adj Close', 'OBV', '10_SMA', '20_SMA', '12_EMA', '26_EMA', 'RSI', 'UpperBB', 'LowerBB', '%K', '%D', 'ADX']])

    # Make predictions
    predictions = []
    current_batch = scaled_data[-60:].reshape(1, 60, 12)
    
    prediction_dates = []
    next_date = input_date
    while len(prediction_dates) < 15:
        next_date += timedelta(days=1)
        if next_date.weekday() < 5:  
            prediction_dates.append(next_date)

    # Output the predictions with buy/sell/hold indicator
    for i in range(15):  
        next_prediction = model.predict(current_batch)
        predicted_price = scaler.inverse_transform([[next_prediction[0, 0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])[0, 0]
        current_price = data['Close'].values[-1]
        if predicted_price > current_price:
            indicator = 'Buy'
        elif predicted_price < current_price:
            indicator = 'Sell'
        else:
            indicator = 'Hold'
        predictions.append([prediction_dates[i].strftime('%Y-%m-%d'), predicted_price, indicator])

        # Prepare for the next prediction
        next_prediction_reshaped = np.array([[next_prediction[0, 0], 
                                              data['OBV'].values[-1],
                                              data['10_SMA'].values[-1], 
                                              data['20_SMA'].values[-1], 
                                              data['12_EMA'].values[-1], 
                                              data['26_EMA'].values[-1],
                                              data['RSI'].values[-1],
                                              data['UpperBB'].values[-1],
                                              data['LowerBB'].values[-1],
                                              data['%K'].values[-1],
                                              data['%D'].values[-1],
                                              data['ADX'].values[-1]]]).reshape(1, 1, 12)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)

    # Create DataFrame for predictions
    df_predictions = pd.DataFrame(predictions, columns=['Date', 'Predicted Price', 'Indicator'])
    st.dataframe(df_predictions)

def main():
    st.title('Stock Price Predictor')
    company_ticker = st.text_input("Enter Company Ticker (e.g., AAPL for Apple Inc.):")
    input_date = st.date_input("Select a Date to Predict Stock Prices for the Next 15 Days:")
    if st.button('Predict'):
        st.write("Predictions:")
        predict_stock_price(input_date, company_ticker)

if __name__ == "__main__":
    main()
