# pyright: reportMissingImports=false
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# -------------------------------
# CONFIGURATION
# -------------------------------
SEQ_LEN = 60
MODEL_PATH = "gru_model.h5"
SCALER_PATH = "scaler.pkl"

# -------------------------------
# LOAD MODEL AND SCALER
# -------------------------------
@st.cache_resource
def load_resources():
    # Load model safely (no compile to avoid mse deserialization issue)
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)

    # Recompile for inference
    model.compile(optimizer='adam', loss='mse')
    return model, scaler

model, scaler = load_resources()

# -------------------------------
# APP INTERFACE
# -------------------------------
st.title("💹 Bitcoin Price Prediction using GRU Model")
st.write("This app uses your trained GRU model to predict and forecast Bitcoin (BTC) prices.")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file with 'Date' and 'Close' columns", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validate file structure
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("❌ The uploaded file must contain 'Date' and 'Close' columns.")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.reset_index(drop=True, inplace=True)

        st.subheader("📄 Data Preview")
        st.dataframe(df.tail())

        # -------------------------------
        # PREPROCESSING
        # -------------------------------
        df['log_close'] = np.log(df['Close'])
        data = df['log_close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(data)

        # Create input sequences
        X = []
        for i in range(SEQ_LEN, len(scaled_data)):
            X.append(scaled_data[i-SEQ_LEN:i])
        X = np.array(X)

        # -------------------------------
        # PREDICTIONS
        # -------------------------------
        preds_scaled = model.predict(X)
        preds = scaler.inverse_transform(preds_scaled)
        preds_price = np.exp(preds)

        actual_price = np.exp(scaler.inverse_transform(scaled_data[SEQ_LEN:]))

        # -------------------------------
        # VISUALIZATION — ACTUAL vs PREDICTED
        # -------------------------------
        st.subheader("📈 Actual vs Predicted BTC Prices")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df['Date'][SEQ_LEN:], actual_price, label='Actual Price', color='blue')
        ax.plot(df['Date'][SEQ_LEN:], preds_price, label='Predicted Price', color='red', linestyle='--')
        ax.set_title("GRU — Actual vs Predicted BTC Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("BTC Price (USD)")
        ax.legend()
        st.pyplot(fig)

        # -------------------------------
        # FORECAST FUTURE BTC PRICES
        # -------------------------------
        st.subheader("🔮 Forecast Future BTC Prices")

        days = st.slider("Select number of days to forecast", 1, 60, 10)
        last_seq = scaled_data[-SEQ_LEN:]
        forecast_scaled = []
        seq = last_seq.copy()

        for _ in range(days):
            pred = model.predict(seq.reshape(1, SEQ_LEN, 1))[0][0]
            forecast_scaled.append(pred)
            seq = np.append(seq[1:], pred).reshape(SEQ_LEN, 1)

        forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
        forecast_log = scaler.inverse_transform(forecast_scaled)
        forecast_price = np.exp(forecast_log)

        # Future date index
        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=days+1, freq='D')[1:]
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast_Price': forecast_price.flatten()})

        st.write(forecast_df)

        # Plot forecast
        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.plot(df['Date'], df['Close'], label='Historical', color='blue')
        ax2.plot(forecast_df['Date'], forecast_df['Forecast_Price'], label='Forecast', color='orange', linestyle='--')
        ax2.set_title(f"BTC Price Forecast — Next {days} Days")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("BTC Price (USD)")
        ax2.legend()
        st.pyplot(fig2)

else:
    st.info("📥 Please upload a CSV file with 'Date' and 'Close' columns to start.")