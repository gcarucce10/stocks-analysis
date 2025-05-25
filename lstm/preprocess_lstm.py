import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss

def check_stationarity(series: pd.Series):
    """
    Applies ADF and KPSS tests to check for stationarity.

    :param series: Time series to test (e.g., closing prices).
    :return: Dictionary with test statistics and p-values.
    """
    adf_result = adfuller(series.dropna())
    kpss_result, kpss_pvalue, _, _ = kpss(series.dropna(), regression='c')

    return {
        "ADF Statistic": adf_result[0],
        "ADF p-value": adf_result[1],
        "KPSS Statistic": kpss_result,
        "KPSS p-value": kpss_pvalue
    }

def preprocess_lstm(df: pd.DataFrame, window_size: int = 6):
    """

    Prepares monthly data for LSTM training.
    
    :param df: DataFrame with the 'close' column and datetime index.
    :param window_size: Number of previous months to use to predict the next one.
    :return: tuple (X, y, scaler) for LSTM training.
    """
    
    df = df.copy()
    df = df[["close"]].dropna()
    
    # Check stationarity
    transform_info = {"log": False, "diff": False}
    stationarity = check_stationarity(df["close"])
    print("\nStationarity Test Results (Preprocessing):")
    for key, value in stationarity.items():
        print(f"{key}: {value}")
        
    # Apply log transform and differencing if non-stationary
    if stationarity["ADF p-value"] > 0.05 or stationarity["KPSS p-value"] < 0.05:
        print("Applying log transformation and differencing for stationarity...")
        df["close"] = np.log(df["close"])
        transform_info["log"] = True
        
        df["close"] = df["close"].diff().dropna()
        df.dropna(inplace=True)
        transform_info["diff"] = True
        
    df.dropna(inplace=True)
        
    # Normalize the closing price data
    scaler = MinMaxScaler()
    df["close_scaled"] = scaler.fit_transform(df[["close"]])
    
    # Create input windows and the corresponding target value
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df["close_scaled"].values[i - window_size:i])
        y.append(df["close_scaled"].values[i])

    # Convert to numpy arrays and adjust shape
    X = np.array(X)
    y = np.array(y)

    # Reshape to (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, transform_info
