import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def linear_regression_stock():
    """
    Function to perform linear regression analysis.
    """
    
    df_mensal = pd.read_csv("/home/useradd/stocks-analysis/data/dados_mensais.csv", index_col="Date", parse_dates=True)

    # Monthly percentage return
    df_mensal["retorno"] = df_mensal["close"].pct_change()

    # Moving averages (3-month and 6-month)
    df_mensal["mm_3"] = df_mensal["close"].rolling(window=3).mean()
    df_mensal["mm_6"] = df_mensal["close"].rolling(window=6).mean()

    # Volatility (3-month standard deviation)
    df_mensal["vol_3"] = df_mensal["close"].rolling(window=3).std()

    # Lag 1 (1-month lag)
    df_mensal["close_lag1"] = df_mensal["close"].shift(1)

    # Target: next month's closing price
    df_mensal["target"] = df_mensal["close"].shift(-1)

    # Remove resulting NaNs
    df_mensal = df_mensal.dropna()

    # Features and target
    X = df_mensal[["retorno", "mm_3", "mm_6", "vol_3", "close_lag1"]]
    y = df_mensal["target"]
    
    df_mensal_predicted = df_mensal.copy()
    
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    # Predict the target variable
    y_pred = model.predict(X_test)

    # Create a new column with NaN values in df_mensal_predicted
    df_mensal_predicted["predicted"] = float("nan")
        
    # Assign predictions to the correct rows (the test set) in the new dataframe
    df_mensal_predicted.loc[X_test.index, "predicted"] = y_pred

    # Drop all columns except 'close', 'predicted' and keep the index (Date)
    df_mensal_predicted = df_mensal_predicted.drop(columns=[col for col in df_mensal_predicted.columns if col not in ['close', 'predicted']])

    # Print the DataFrame with only the desired columns
    print(df_mensal_predicted)
    
    df_mensal_predicted.to_csv("/home/useradd/stocks-analysis/data/dados_mensais_predicted_linear_regression.csv", index=True)
    
