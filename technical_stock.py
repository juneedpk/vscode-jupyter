
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import load_model
import mplfinance as mpf
from matplotlib.dates import DateFormatter
import matplotlib as mpl

# Function to calculate MACD and Signal Line
def calculate_macd(df):
    ShortEMA = df['Close'].ewm(span=12, adjust=False).mean()
    LongEMA = df['Close'].ewm(span=26, adjust=False).mean()
    MACD = ShortEMA - LongEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    return MACD, signal

# Function to generate buy and sell signals based on MACD and Signal Line
def buy_sell_macd(signal):
    Buy = []
    Sell = []
    flag = -1

    for i in range(len(signal)):
        if signal['MACD'][i] > signal['Signal Line'][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(signal['Close'][i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif signal['MACD'][i] < signal['Signal Line'][i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(signal['Close'][i])
                flag = 0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    return Buy, Sell

# Function to calculate buy and sell signals based on EWMA
def buy_sell_ewma3(data):
    buy_list = []
    sell_list = []
    flag_long = False
    flag_short = False

    for i in range(0, len(data)):
        if data['Middle'][i] < data['Long'][i] and data['Short'][i] < data['Middle'][i] and flag_short == False and flag_long == False:
            buy_list.append(data['Close'][i])
            sell_list.append(np.nan)
            flag_short = True
        elif flag_short == True and data['Short'][i] > data['Middle'][i]:
            sell_list.append(data['Close'][i])
            buy_list.append(np.nan)
            flag_short = False
        elif data['Middle'][i] > data['Long'][i] and data['Short'][i] > data['Middle'][i] and flag_short == False and flag_long == False:
            buy_list.append(data['Close'][i])
            sell_list.append(np.nan)
            flag_long = True
        elif flag_long == True and data['Short'][i] < data['Middle'][i]:
            sell_list.append(data['Close'][i])
            buy_list.append(np.nan)
            flag_long = False
        else:
            buy_list.append(np.nan)
            sell_list.append(np.nan)

    return buy_list, sell_list

# Function to plot buy and sell signals for EWMA
def buy_sell_ewma3_plot(df):
    sns.set(rc={'figure.figsize': (18, 10)})
    plt.plot(df['Close'], label=f"Close", color='blue', alpha=0.35)
    plt.plot(df['Short'], label='Short/Fast EMA', color='red', alpha=0.35)
    plt.plot(df['Middle'], label='Middle/Medium EMA', color='orange', alpha=0.35)
    plt.plot(df['Long'], label='Long/Slow EMA', color='green', alpha=0.35)
    plt.scatter(df.index, df['Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
    plt.scatter(df.index, df['Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
    plt.title("EWMA Buy/Sell Signals", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Stock Price (p)', color='black', fontsize=15)
    plt.legend()
    st.pyplot(plt)

# Function to calculate RSI
def calculate_rsi(df, period=14):
    delta = df['Close'].diff(1)
    delta = delta.dropna()
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    AVG_Gain = up.rolling(window=period).mean()
    AVG_Loss = down.abs().rolling(window=period).mean()
    RS = AVG_Gain / AVG_Loss
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

# Function to plot RSI
def plot_rsi(df, rsi):
    df = df.iloc[1:]  # Drop the first row to match the length of RSI
    plt.figure(figsize=(20, 10))
    plt.plot(df.index, rsi, label='RSI')
    plt.axhline(30, linestyle='--', alpha=0.5, color='red')
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')
    plt.title('RSI', color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('RSI', color='black', fontsize=15)
    plt.legend(loc='upper left')
    st.pyplot(plt)

# Function to calculate Bollinger Bands
def calculate_bb(df, period=20):
    df['SMA'] = df['Close'].rolling(window=period).mean()
    df['STD'] = df['Close'].rolling(window=period).std()
    df['Upper'] = df['SMA'] + (df['STD'] * 2)
    df['Lower'] = df['SMA'] - (df['STD'] * 2)
    return df

# Function to get the buy and sell signals for Bollinger Bands
def get_signal_bb(data):
    buy_signal = []  # buy list
    sell_signal = []  # sell list

    for i in range(len(data['Close'])):
        if data['Close'][i] > data['Upper'][i]:  # Then you should sell
            buy_signal.append(np.nan)
            sell_signal.append(data['Close'][i])
        elif data['Close'][i] < data['Lower'][i]:  # Then you should buy
            sell_signal.append(np.nan)
            buy_signal.append(data['Close'][i])
        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)
    return buy_signal, sell_signal

# Function to plot Bollinger Bands
def bb_shaded(df):
    fig = plt.figure(figsize=(20, 10))  # Get the figure and the figure size
    ax = fig.add_subplot(1, 1, 1)  # Add the subplot

    x_axis = df.index  # Get the index values of the DataFrame
    ax.fill_between(x_axis, df['Upper'], df['Lower'], color='grey')  # Plot and shade the area between the upper and lower band

    ax.plot(x_axis, df['Close'], color='gold', lw=3, label='Close Price')  # Plot the Closing Price
    ax.plot(x_axis, df['SMA'], color='blue', lw=3, label='Simple Moving Average')  # Plot the Moving Average

    ax.set_title('Bollinger Band Stock', color='black', fontsize=20)  # Set the Title & Show the Image
    ax.set_xlabel('Date', color='black', fontsize=15)
    ax.set_ylabel('Close Price', color='black', fontsize=15)
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(plt)

# Function to plot Bollinger Bands with buy and sell signals
def bb_alldata(df):
    fig = plt.figure(figsize=(20, 10))  # Get the figure and the figure size
    ax = fig.add_subplot(1, 1, 1)  # Add the subplot

    x_axis = df.index  # Get the index values of the DataFrame
    ax.fill_between(x_axis, df['Upper'], df['Lower'], color='grey')  # Plot and shade the area between the upper and lower band

    ax.plot(x_axis, df['Close'], color='gold', lw=3, label='Close Price', alpha=0.5)  # Plot the Closing Price
    ax.plot(x_axis, df['SMA'], color='blue', lw=3, label='Moving Average', alpha=0.5)  # Plot the Moving Average
    ax.scatter(x_axis, df['Buy'], color='green', lw=3, label='Buy', marker='^', alpha=1)  # Plot Buy signals
    ax.scatter(x_axis, df['Sell'], color='red', lw=3, label='Sell', marker='v', alpha=1)  # Plot Sell signals

    ax.set_title('Bollinger Band, Close Price, MA and Trading Signals', color='black', fontsize=20)  # Set the Title & Show the Image
    ax.set_xlabel('Date', color='black', fontsize=15)
    ax.set_ylabel('Close Price', color='black', fontsize=15)
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(plt)

# Function to calculate ROC
def calculate_roc(df, period=12):
    N = df['Close'].diff(period)
    D = df['Close'].shift(period)
    ROC = pd.Series(N / D, name='ROC')
    df = df.join(ROC)
    return df

# Function to plot ROC with custom styling
def plot_roc(df, roc_period):
    dates = df.index
    price = df['Close']
    roc = df['ROC']

    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(hspace=0)
    plt.rcParams.update({'font.size': 14})

    # Price subplot
    price_ax = plt.subplot(2, 1, 1)
    price_ax.plot(dates, price, color='blue', linewidth=2, label="Adj Closing Price")
    price_ax.legend(loc="upper left", fontsize=12)
    price_ax.set_ylabel("Price")
    price_ax.set_title("Daily Price", fontsize=24)

    # ROC subplot
    roc_ax = plt.subplot(2, 1, 2, sharex=price_ax)
    roc_ax.plot(roc, color='k', linewidth=1, alpha=0.7, label="9-Day ROC")
    roc_ax.legend(loc="upper left", fontsize=12)
    roc_ax.set_ylabel("% ROC")

    # Adding a horizontal line at the zero level in the ROC subplot:
    roc_ax.axhline(0, color=(.5, .5, .5), linestyle='--', alpha=0.5)

    # Filling the areas between the indicator and the level 0 line:
    roc_ax.fill_between(dates, 0, roc, where=(roc >= 0), color='g', alpha=0.3, interpolate=True)
    roc_ax.fill_between(dates, 0, roc, where=(roc < 0), color='r', alpha=0.3, interpolate=True)

    # Formatting the date labels
    roc_ax.xaxis.set_major_formatter(DateFormatter('%b'))

    # Formatting the labels on the y axis for ROC:
    roc_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())

    # Adding a grid to both subplots:
    price_ax.grid(True, linestyle='--', alpha=0.5)
    roc_ax.grid(True, linestyle='--', alpha=0.5)

    # Setting a background color for both subplots:
    price_ax.set_facecolor((.94, .95, .98))
    roc_ax.set_facecolor((.98, .97, .93))

    # Adding margins around the plots:
    price_ax.margins(0.05, 0.2)
    roc_ax.margins(0.05, 0.2)

    # Hiding the tick marks from the horizontal and vertical axis:
    price_ax.tick_params(left=False, bottom=False)
    roc_ax.tick_params(left=False, bottom=False, labelrotation=45)

    # Hiding all the spines for the price subplot:
    for s in price_ax.spines.values():
        s.set_visible(False)
    # Hiding all the spines for the ROC subplot:
    for s in roc_ax.spines.values():
        s.set_visible(False)

    # To better separate the two subplots, we reinstate a spine in between them
    roc_ax.spines['top'].set_visible(True)
    roc_ax.spines['top'].set_linewidth(1.5)

    st.pyplot(plt)

# Streamlit app layout
st.title("Technical Analysis Dashboard")

# Sidebar options for stock ticker and date range
st.sidebar.title("Options")
ticker = st.sidebar.text_input('Enter Stock Ticker', value='AAPL')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('today'))

# Select the analysis to perform
option = st.sidebar.selectbox(
    'Select the analysis to perform:',
    ('MACD', 'EWMA', 'RSI', 'Bollinger Bands', 'ROC')
)

# Load data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_data(ticker, start_date, end_date)

if option == 'MACD':
    data['MACD'], data['Signal Line'] = calculate_macd(data)
    data['Buy'], data['Sell'] = buy_sell_macd(data)

    st.subheader('MACD & Signal Line')
    plt.figure(figsize=(20, 10))
    plt.plot(data['MACD'], label='MACD', color='blue')
    plt.plot(data['Signal Line'], label='Signal Line', color='red')
    plt.legend(loc='upper left')
    st.pyplot(plt)

    st.subheader('Buy and Sell Signals')
    plt.figure(figsize=(20, 10))
    plt.plot(data['Close'], label='Close Price', alpha=0.5)
    plt.scatter(data.index, data['Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
    plt.scatter(data.index, data['Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
    plt.title('Buy and Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left')
    st.pyplot(plt)

    st.subheader('DataFrame with Buy/Sell Signals')
    st.dataframe(data[['Close', 'MACD', 'Signal Line', 'Buy', 'Sell']])

elif option == 'EWMA':
    short_ewma = data['Close'].ewm(span=12, adjust=False).mean()
    middle_ewma = data['Close'].ewm(span=26, adjust=False).mean()
    long_ewma = data['Close'].ewm(span=50, adjust=False).mean()
    data['Short'] = short_ewma
    data['Middle'] = middle_ewma
    data['Long'] = long_ewma
    data['Buy'], data['Sell'] = buy_sell_ewma3(data)

    st.subheader('EWMA Buy/Sell Signals')
    buy_sell_ewma3_plot(data)

    st.subheader('DataFrame with Buy/Sell Signals')
    st.dataframe(data[['Close', 'Short', 'Middle', 'Long', 'Buy', 'Sell']])

elif option == 'RSI':
    rsi = calculate_rsi(data)
    data['RSI'] = rsi

    st.subheader('RSI')
    plot_rsi(data, rsi)

elif option == 'Bollinger Bands':
    azn_12mo_bb = calculate_bb(data)
    azn_12mo_bb['Buy'], azn_12mo_bb['Sell'] = get_signal_bb(azn_12mo_bb)
    
    st.subheader('Bollinger Bands')
    bb_shaded(azn_12mo_bb)
    
    st.subheader('Bollinger Bands with Buy/Sell Signals')
    bb_alldata(azn_12mo_bb)

    st.subheader('DataFrame with Buy/Sell Signals')
    st.dataframe(azn_12mo_bb[['Close', 'SMA', 'Upper', 'Lower', 'Buy', 'Sell']])

elif option == 'ROC':
    roc_period = 12
    data = calculate_roc(data, roc_period)
    st.subheader(f'Rate of Change (ROC) - {roc_period} Days')
    plot_roc(data, roc_period)
