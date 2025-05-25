import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm.preprocess_lstm import preprocess_lstm

def train_and_predict_lstm(window_size: int = 6):
    """
    Trains an LSTM model on the provided DataFrame and returns predictions.
    
    :param window_size: Number of previous months to use to predict the next one.
    """
    
    df_mensal = pd.read_csv("/home/useradd/stocks-analysis/data/dados_mensais.csv", index_col="Date", parse_dates=True)
    
    # Preprocess the data
    X, y, scaler, transform_info = preprocess_lstm(df_mensal, window_size)
    
    # Split into training and test sets (leave the last sample for prediction)
    X_train, y_train = X[:-1], y[:-1]
    last_window = X[-1].reshape(1, window_size, 1)
        
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    
    # Predict the next month using the last available window
    next_month_scaled = model.predict(last_window)[0][0]
    
    # Invert normalization
    predicted_scaled_series = pd.Series([next_month_scaled])
    predicted_transformed = scaler.inverse_transform(predicted_scaled_series.values.reshape(-1, 1))[0][0]
    
    # Invert log transformation and differencing if applied
    if transform_info.get("diff"):
        last_transformed_price = df_mensal["close"].copy()
        if transform_info.get("log"):
            last_transformed_price = np.log(last_transformed_price)
        last_transformed_price = last_transformed_price.dropna().iloc[-1]
        predicted_transformed += last_transformed_price
    
    if transform_info.get("log"):
        predicted_transformed = np.exp(predicted_transformed)
        
    next_month_value = predicted_transformed
    
    # Prepare output DataFrame
    df_result = df_mensal.copy()
    df_result = df_result["close"].reset_index()
    df_result.rename(columns={"close": "real"}, inplace=True)
    df_result["predicted"] = np.nan
    
    # Append prediction for next month
    last_date = df_result["Date"].iloc[-1]
    next_date = last_date + pd.DateOffset(months=1)
    predicted_row = pd.DataFrame({
        "Date": [next_date],
        "real": [np.nan],  # real value not yet known
        "predicted": [next_month_value]
    })
    
    df_result = pd.concat([df_result, predicted_row], ignore_index=True)

    df_result.to_csv("/home/useradd/stocks-analysis/data/dados_mensais_predicted_lstm.csv", index=False)
    print(df_result)
    
def evaluate_lstm_model(window_size: int = 6):
    """
    Evaluate LSTM model performance on historical data using MAE, RMSE, and MSE.

    :param window_size: Number of months to use as input for prediction.
    """
    df_mensal = pd.read_csv("/home/useradd/stocks-analysis/data/dados_mensais.csv", index_col="Date", parse_dates=True)
    df_original = df_mensal.copy()
    
    X, y, scaler, transform_info = preprocess_lstm(df_mensal, window_size=window_size)

    # Define and train the model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)
    
    # Predict and inverse transform
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    
    # Inverse transformations if applied 
    if transform_info.get("diff"):
        base_series = np.log(df_original["close"]) if transform_info.get("log") else df_original["close"]
        base_series = base_series.dropna().values
        base_for_reconstruction = base_series[window_size:-1]  # alinhamento correto
        y_pred += base_for_reconstruction
        y_true += base_for_reconstruction

    if transform_info.get("log"):
        y_pred = np.exp(y_pred)
        y_true = np.exp(y_true)
        
    # Avaliation metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print("\nEvaluating the model on historical data:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE:  {mse:.2f}")
    
    # Save results to CSV
    dates = df_original.index[-len(y_true):]
    df_result = pd.DataFrame({
        "Date": dates,
        "real": y_true,
        "predicted": y_pred
    })
    df_result.to_csv("/home/useradd/stocks-analysis/data/dados_mensais_evaluation_lstm.csv", index=False)
    print(df_result)  