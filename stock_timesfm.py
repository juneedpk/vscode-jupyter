
#huggingface-cli login --token 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

import datetime
import yfinance as yf
import pandas as pd
from timesfm import TimesFm

# Function to get top 100 NASDAQ stocks from CSV file
def get_top_nasdaq_stocks(csv_file, n=100):
    nasdaq_df = pd.read_csv(csv_file)
    top_stocks = nasdaq_df['Ticker'].tolist()[:n]
    return top_stocks

# Path to your CSV file
csv_file = 'https://raw.githubusercontent.com/juneedpk/vscode-jupyter/master/nasdaq100.csv'

# Dates for data fetching
start = datetime.date(2022, 1, 1)
end = datetime.date.today()
codelist = get_top_nasdaq_stocks(csv_file)

# Data fetching
data_dict = {}
for code in codelist:
    data = yf.download(code, start=start, end=end)
    if not data.empty:
        data_dict[code] = data['Adj Close'].dropna()

context_len = 512  # Setting maximum context length
horizon_len = 10  # Set the length of the forecast period

# Initialize and load TimesFM model
tfm = TimesFm(
    context_len=context_len,
    horizon_len=horizon_len,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend='cpu',
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# Initialize lists to store predictions and alerts
predictions = []
alerts = []

# Process each stock
for code, data in data_dict.items():
    available_data_len = len(data)
    if available_data_len < context_len:
        adjusted_context_len = available_data_len
        print(f"Adjusting context length for {code} to {adjusted_context_len}")
    else:
        adjusted_context_len = context_len

    context_data = data[-adjusted_context_len:]  # Use the available days of data as context

    # Prepare data
    forecast_input = [context_data.values]
    frequency_input = [0]  # Set data frequency (1 is daily frequency data)

    # Run prediction
    point_forecast, experimental_quantile_forecast = tfm.forecast(
        forecast_input,
        freq=frequency_input,
    )

    # Display prediction results
    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=horizon_len, freq='B')
    forecast_series = pd.Series(point_forecast[0], index=forecast_dates)

    # Append predictions to the list
    predictions.append((code, forecast_series))

    # Check for price drop alert
    if forecast_series.pct_change().min() < -0.03:
        alerts.append((code, forecast_series.pct_change().min()))

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame({code: forecast for code, forecast in predictions})

# Convert alerts to a DataFrame
alerts_df = pd.DataFrame(alerts, columns=['Stock', 'Max Drop'])

# Display the DataFrames
print("NASDAQ 100 Predictions:")
print(predictions_df)

print("\nStocks with Alerts (predicted to drop more than 3%):")
print(alerts_df)