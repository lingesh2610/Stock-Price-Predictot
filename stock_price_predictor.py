import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as download_data # Library to fetch real stock data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

def fetch_stock_data(ticker, years=5):
    """Fetches historical data using yfinance."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = download_data.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError("Could not fetch data. Check the ticker symbol or internet connection.")
    return data

def prepare_data(data, window_size=5):
    """
    Creates 'features' based on past prices (Windowing).
    Predicts today's price based on the previous 5 days.
    """
    df = data[['Close']].copy()
    
    # Create features: prices from the last 'n' days
    for i in range(1, window_size + 1):
        df[f'Prev_{i}'] = df['Close'].shift(i)
        
    # Drop rows with NaN values created by shifting
    df.dropna(inplace=True)
    
    X = df.drop('Close', axis=1)
    y = df['Close']
    
    return X, y, df

def train_and_predict(ticker_symbol="AAPL"):
    try:
        # 1. Load Data
        raw_data = fetch_stock_data(ticker_symbol)
        
        # 2. Preprocess
        X, y, df_processed = prepare_data(raw_data)
        
        # 3. Split into Train/Test (80/20)
        # Note: In time-series, we usually don't shuffle, but Random Forest handles this well
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # 4. Initialize and Train Model
        # Using Random Forest Regressor for better non-linear capturing than Linear Regression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 5. Make Predictions
        predictions = model.predict(X_test)
        
        # 6. Evaluation Metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        print("\n" + "="*30)
        print(f"MODEL PERFORMANCE ({ticker_symbol})")
        print("="*30)
        print(f"Root Mean Squared Error: ${rmse:.2f}")
        print(f"R-Squared Score: {r2:.4f}")
        
        # 7. Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test.values, label='Actual Price', color='blue', linewidth=2)
        plt.plot(y_test.index, predictions, label='Predicted Price', color='red', linestyle='--', linewidth=2)
        plt.title(f'{ticker_symbol} Stock Price Prediction (Test Set)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 8. Predict Tomorrow's Price
        last_features = X.iloc[-1].values.reshape(1, -1)
        tomorrow_pred = model.predict(last_features)[0]
        print(f"\nEstimated Price for Next Trading Day: ${tomorrow_pred:.2f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # You can change 'AAPL' to 'TSLA', 'GOOGL', 'MSFT', etc.
    symbol = input("Enter a Stock Ticker (e.g., AAPL, TSLA, MSFT): ").upper() or "AAPL"
    train_and_predict(symbol)
    