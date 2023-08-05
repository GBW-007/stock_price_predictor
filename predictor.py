import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from ta import add_all_ta_features

# Function to fetch historical stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to create additional features from the historical stock data
def create_features(data):
    data['Date'] = data.index
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    # Add technical indicators
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")

    # Add lagged features (past price movements)
    num_lagged_days = 5
    for i in range(1, num_lagged_days + 1):
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)

    return data

# Function to prepare the dataset for modeling
def prepare_dataset(data):
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek',
        'momentum_rsi',  # Relative Strength Index (RSI)
        'trend_sma_fast',  # Simple Moving Average (SMA) as a fast trend indicator
        'trend_sma_slow',  # Simple Moving Average (SMA) as a slow trend indicator
        'volume_adi',  # Accumulation/Distribution Index
        'volatility_bbm',  # Bollinger Bands Mid Band
        'volatility_bbw',  # Bollinger Bands Width
    ]
    for i in range(1, num_lagged_days + 1):
        features.append(f'Close_Lag_{i}')

    X = data[features].values
    y = data['Close'].shift(-1).dropna().values  # Using tomorrow's closing price as the target

    # Handle missing values (NaN) with SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X, y

# Main function to fetch data, create features, build and evaluate the model
def predict_stock_price(ticker, start_date, end_date):
    # Fetching data from Yahoo Finance
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Creating additional features (including technical indicators and lagged features)
    stock_data = create_features(stock_data)

    # Preparing dataset for modeling
    X, y = prepare_dataset(stock_data)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y, test_size=0.2, random_state=42)

    # Creating and training the Gradient Boosting Regressor model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Predicting the stock price for a future date (tomorrow)
    last_row = stock_data.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek',
                                    'momentum_rsi', 'trend_sma_fast', 'trend_sma_slow',
                                    'volume_adi', 'volatility_bbm', 'volatility_bbw'] +
                                    [f'Close_Lag_{i}' for i in range(1, num_lagged_days + 1)]].values.reshape(1, -1)
    tomorrow_prediction = model.predict(last_row)
    print(f"Predicted stock price for tomorrow: {tomorrow_prediction[0]}")

if __name__ == "__main__":
    ticker = "TSLA"  # Replace with the stock symbol you want to predict
    start_date = "2020-01-01"
    end_date = "2023-08-05"  # Replace with the current date or your desired end date
    num_lagged_days = 5  # Number of lagged (past price) features to include
    predict_stock_price(ticker, start_date, end_date)

