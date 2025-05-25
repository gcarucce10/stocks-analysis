import yfinance as yf
import pandas as pd

def baixar_dados(ticker: str, anos: int = 10) -> pd.DataFrame:
    # Download historical stock data using yfinance
    # ticker: stock symbol (e.g., "AAPL" for Apple)
    # anos: number of years of data to download
    # Returns a DataFrame with daily closing data
    df = yf.Ticker(ticker).history(period=f"{anos}y", interval="1d")
    # Filter only the daily closing column
    df = df[['Close']].dropna()
    # Rename the column to 'close'
    df.columns = ['close']
    return df

# Save the DataFrame to a CSV file
def salvar_csv(df: pd.DataFrame, caminho: str = "data/dados_diarios.csv"):
    df.to_csv(caminho)

# Function to aggregate daily data into monthly data
# The function receives a DataFrame with daily data and returns a DataFrame with monthly data
def agregacao_mensal(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df_mensal = df.resample('M').last()
    return df_mensal

