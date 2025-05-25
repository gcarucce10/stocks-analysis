from src.preprocess import baixar_dados, salvar_csv, agregacao_mensal
from plots.plots import plot_monthly_closing_comparison_linear_regression, plot_monthly_closing_comparison_random_forest
from lstm.pipeline_lstm import train_and_predict_lstm, evaluate_lstm_model
import pandas as pd

if __name__ == "__main__":
    # Download data from Yahoo Finance API and save as CSV (daily data) for the last 10 years
    df = baixar_dados("AAPL", 10)
    salvar_csv(df, "data/dados_diarios.csv")

    # Group daily data into monthly data (month-end closing) and save as CSV
    df_mensal = agregacao_mensal(df)
    salvar_csv(df_mensal, "data/dados_mensais.csv")
    
    # Linear regression method
    # linear_regression_stock()
    
    # Random forest method
    # random_forest_stock()
    
    # Plotting the difference between the real and predicted monthly closing prices using linear regression
    # plot_monthly_closing_comparison_linear_regression()
    
    # Plotting the difference between the real and predicted monthly closing prices using random forest
    # plot_monthly_closing_comparison_random_forest()
    
    # LSTM method
    train_and_predict_lstm()
    evaluate_lstm_model()
