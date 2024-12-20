from fastapi import FastAPI
import pandas as pd
import yfinance as yf
import joblib
import numpy as np

import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stock-trend-predictor-vqs4.onrender.com"],  # Replace '*' with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "stock_prediction_model.pkl")
model = joblib.load(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI(title="Stock Movement Prediction API")

# Function to fetch and preprocess data
def fetch_and_preprocess_data(ticker):
    try:
        
        # Fetch latest stock data (last 30 days for indicators)
        data = yf.download(ticker, period="1mo", interval="1d")
    
        # Feature Engineering (recalculate indicators)
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['RSI'] = compute_rsi(data['Close'], window=14)
        data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
        data['BB_High'] = data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std()
        data['BB_Low'] = data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std()
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        
        data = data.dropna()
        # Take the most recent row for prediction
        latest_data = data.iloc[-1]
        
        # Return only the required features
        features = [
            latest_data['SMA_10'], latest_data['EMA_10'], latest_data['RSI'],
            latest_data['MACD'], latest_data['Signal_Line'],
            latest_data['BB_High'], latest_data['BB_Low'], latest_data['VWAP']
        ]
        return np.array(features).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Error fetching or processing data: {e}")

# Function to calculate RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# API Endpoint for Prediction
@app.get("/predict/")
def predict_stock(ticker: str):
    """
    Predict whether the stock will go Up or Down based on the latest data.
    """
    try:
        # Fetch and preprocess data
        features = fetch_and_preprocess_data(ticker)
        
        # Make prediction
        prediction_proba = model.predict_proba(features)[0][1]  # Probability for 'Up'
        prediction = "Up" if prediction_proba > 0.5 else "Down"
        
        # Return result
        return {
            "ticker": ticker.upper(),
            "prediction": prediction,
            "confidence_percentage": f"{prediction_proba * 100:.2f}%"
        }
    except ValueError as e:
        return {"error": str(e)}
