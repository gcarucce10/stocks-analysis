import pandas as pd

def media_mediana(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a média e mediana dos preços de fechamento mensal.
    
    :param df: DataFrame com os dados mensais.
    :return: DataFrame com a média dos preços de fechamento mensal.
    """
    df['close'] = df['close'].astype(float)
    media_mensal = df.groupby(df.index.month)['close'].mean()
    mediana_mensal = df.groupby(df.index.month)['close'].median()
    print(media_mensal)
    print(mediana_mensal)
    return media_mensal, mediana_mensal

def minimo_maximo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o preço mínimo e máximo dos preços de fechamento mensal.
    
    :param df: DataFrame com os dados mensais.
    :return: DataFrame com o preço mínimo e máximo dos preços de fechamento mensal.
    """
    df['close'] = df['close'].astype(float)
    minimo_mensal = df.groupby(df.index.month)['close'].min()
    maximo_mensal = df.groupby(df.index.month)['close'].max()
    print(minimo_mensal)
    print(maximo_mensal)
    return minimo_mensal, maximo_mensal

def desvio_padrao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o desvio padrão dos preços de fechamento mensal.
    
    :param df: DataFrame com os dados mensais.
    :return: DataFrame com o desvio padrão dos preços de fechamento mensal.
    """
    df['close'] = df['close'].astype(float)
    desvio_padrao_mensal = df.groupby(df.index.month)['close'].std()
    print(desvio_padrao_mensal)
    return desvio_padrao_mensal

def quantis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os quantis dos preços de fechamento mensal.
    
    :param df: DataFrame com os dados mensais.
    :return: DataFrame com os quantis dos preços de fechamento mensal.
    """
    df.index = pd.to_datetime(df.index)
    df['close'] = df['close'].astype(float)
    quantis_mensal = df.groupby(df.index.month)['close'].quantile([0.25, 0.5, 0.75])
    print(quantis_mensal)
    return quantis_mensal




