import yfinance as yf
import pandas as pd

def baixar_dados(ticker: str, anos: int = 10) -> pd.DataFrame:
    # Baixar dados históricos de ações usando yfinance
    # ticker: símbolo da ação (ex: "AAPL" para Apple)
    # anos: número de anos de dados a serem baixados
    # Retorna um DataFrame com os dados de fechamento diário
    df = yf.Ticker(ticker).history(period=f"{anos}y", interval="1d")
    # Filtra apenas a coluna do fechamento diário
    df = df[['Close']].dropna()
    # Renomeia a coluna para 'close'
    df.columns = ['close']
    return df

# Salvar o DataFrame em um arquivo CSV
def salvar_csv(df: pd.DataFrame, caminho: str = "data/dados_diarios.csv"):
    df.to_csv(caminho)

# Função para agregar os dados diários em mensais
# A função recebe um DataFrame com dados diários e retorna um DataFrame com dados mensais
def agregacao_mensal(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df_mensal = df.resample('M').last()
    return df_mensal

