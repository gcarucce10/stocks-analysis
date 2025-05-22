# 📈 Stock Price Prediction with Machine Learning

This project aims to apply **data science** and **machine learning** and a **deep learning** techniques to forecast the **monthly closing price** of a stock based on historical data from the past 10 years.

---

## 🚀 Goal

Predict the **next month's closing price** of a stock (e.g., AAPL), using time series techniques and regression models.

---

## 🧰 Technologies Used

- Python 3.8  
- Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn  
- Yahoo Finance (via `yfinance`)  
- Git + GitHub  

---

## 📉 Analysis Pipeline

1. **Data Collection**  
   - Daily stock data collected via `yfinance`  
   - Saved to `data/dados_diarios.csv`

2. **Monthly Aggregation**  
   - Converts daily data to monthly frequency  
   - Saved to `data/dados_mensais.csv`

3. **Descriptive Analysis**  
   - Monthly statistics and quantile computation

4. **Modeling (in progress)**  
   - Linear Regression, ARIMA, LSTM (future)

---

## 🔍 Next Steps

- Apply regression models (Linear, Ridge, Random Forest)  
- Test time series models (ARIMA, Prophet, LSTM)  
- Evaluate using MAE, RMSE  
- Optional: Deploy with Streamlit

---

