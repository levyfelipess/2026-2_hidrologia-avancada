import pandas as pd

def head_tail_info(df: pd.DataFrame, n: int = 5):
    """
    Exibe, de uma vez só, as primeiras e as últimas "n" linhas do dataframe, juntamente a informações gerais do mesmo.

    Args:
        df (pd.DataFrame): Dataframe pandas;
        n (int, optional): Número de linhas a serem exibidas ("n" primeiras e "n" últimas).
    """
    display(pd.concat((df.head(n), df.tail(n))), df.info())