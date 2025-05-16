import matplotlib.pyplot as plt
import pandas as pd

# Function to plot real and predicted monthly closing prices
def plot_monthly_closing_comparison_linear_regression():
    # Load the data from CSV files into DataFrames
    # df_mensal: real monthly closing prices
    # df_mensal_predicted_linear_regression: predicted prices from linear regression
    df_mensal = pd.read_csv("/home/useradd/stocks-analysis/data/dados_mensais.csv")
    df_mensal_predicted_linear_regression = pd.read_csv("/home/useradd/stocks-analysis/data/dados_mensais_predicted_linear_regression.csv")

    # Build the plot using real and predicted data
    plt.figure(figsize=(12, 6))
    plt.plot(df_mensal["Date"], df_mensal["close"], label="Actual Closing Price", color="blue")
    plt.plot(df_mensal_predicted_linear_regression["Date"], df_mensal_predicted_linear_regression["predicted"], label="Predicted Closing Price (Linear Regression)", color="red")
    plt.title("AAPL Monthly Closing Price (2015-2025) and Linear Regression Prediction")
    plt.xlabel("Year")
    plt.ylabel("Closing Price ($)")
    plt.legend()
    plt.grid()
    plt.savefig("plots/fechamento_mensal_comparison.png")
