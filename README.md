# 📈 Stock Price Prediction with Machine Learning

This project aims to apply **data science** and **machine learning** techniques to forecast the **monthly closing price** of a stock based on historical data from the past 10 years.

---

## 🚀 Goal

Predict the **next month's closing price** of a stock (e.g., AAPL), using time series techniques and regression models.

---

## 🧰 Technologies Used

- Python 3.8  
- Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn  
- Yahoo Finance (via `yfinance`)  
- Jupyter Notebook  
- Git + GitHub  

---

## 📉 Analysis Pipeline

1. **Data Collection**  
   - Daily stock data collected via `yfinance`  
   - Saved to `data/daily_data.csv`

2. **Monthly Aggregation**  
   - Converts daily data to monthly frequency  
   - Saved to `data/monthly_data.csv`

3. **Descriptive Analysis**  
   - Monthly statistics and quantile computation

4. **Feature Engineering**  
   - Calculates returns, moving averages, volatility, and lag variables  
   - Saved to `data/features_data.csv`

5. **Modeling (in progress)**  
   - Linear Regression, ARIMA, LSTM (future)

---

## 📊 Example of Engineered Features

- `retorno`: monthly percentage return  
- `mm_3`, `mm_6`: moving averages  
- `vol_3`: volatility  
- `close_lag1`: lagged closing price  
- `target`: next month’s closing price (prediction target)

---

## 🔍 Next Steps

- Apply regression models (Linear, Ridge, Random Forest)  
- Test time series models (ARIMA, Prophet, LSTM)  
- Evaluate using MAE, RMSE  
- Optional: Deploy with Streamlit

---

