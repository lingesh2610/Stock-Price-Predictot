# 📈 Stock Price Predictor

## 🚀 Project Overview

This project is an AI-driven stock price forecasting tool developed during a 1-month internship.  
It uses historical market data to predict future closing prices of publicly traded companies.

Unlike simple linear models, this project utilizes **Random Forest Regression** to better account for the volatility and non-linear patterns found in financial markets.

---

## 🧠 Technical Approach

### 📊 Data Acquisition
- Integrated with the **yfinance API**
- Pulls real-time historical data directly from Yahoo Finance

### ⚙️ Feature Engineering
- Implemented a **Windowing Technique**
- Uses the closing prices of the previous **5 trading days**
- These act as input features to predict the current day's closing price

### 🤖 Machine Learning Model
- **Algorithm:** Random Forest Regressor
- **Why Random Forest?**
  - Reduces overfitting
  - Handles non-linear patterns
  - Manages outliers better than Linear Regression
  - More suitable for volatile financial data

### 🧪 Validation Strategy
- Used a **Time-Series Split**
- 80% training data
- 20% testing data
- Ensures the model learns from past data to predict future values

---

## 📉 Performance Metrics

The model is evaluated using:

- **Root Mean Squared Error (RMSE)**  
  Measures the average prediction error in dollars.

- **R-Squared (R²)**  
  Indicates how well the features explain stock price variability (higher is better).

---

## 🛠 Installation

Ensure Python 3.x is installed.

Then install the required packages:

```bash
py -m pip install yfinance scikit-learn matplotlib pandas numpy
```

---

## ▶️ How to Use

Run the script:

```bash
python stock_predictor.py
```

Then:

1. Enter a valid stock ticker symbol  
   - Example: `NVDA` (NVIDIA)  
   - Example: `BTC-USD` (Bitcoin)

2. View the generated graph comparing:
   - Actual Prices
   - Predicted Prices

3. Check the terminal for next-day price estimation.

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**.  
It should not be used for real financial trading or investment decisions.
