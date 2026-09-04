import numpy as np
import pandas as pd

def build_timeseries_df_from_daily_avg_pr(avg_pr_array: np.ndarray, date_start: str, date_end: str,
                                          missing_pr_value: float | None = -999.) -> pd.DataFrame:
    """
    Constrói um dataframe de série temporal de precipitações médias diárias a partir de um array.

    Args:
        avg_pr_array (np.ndarray[n, 1] or
                      np.ndarray[1, n] or
                      np.ndarray[n]): vetor de precipitações médias diárias;
        missing_pr_value (float or None, optional): Valor numérico correspondente às precipitações faltantes. Se "None", não substitui;
        date_start (str): Data do dia inicial no formato 'AAAA-MM-DD';
        date_end (str): Data do dia final no formato 'AAAA-MM-DD';

    Returns:
        pd.DataFrame[n, 2]: Série temporal como dataframe de "n" dias:
            Coluna 0 ('date'): Índices temporais da série no formato 'AAAA-MM-DD';
            Coluna 1 ('avg_pr'): Precipitações médias diárias.
    """
    if missing_pr_value is not None:
        avg_pr_array = np.where(avg_pr_array == missing_pr_value, np.nan, avg_pr_array)

    df = pd.DataFrame({
        'date':pd.date_range(start=date_start, end=date_end, freq='D'),
        'avg_pr':np.reshape(avg_pr_array, -1)
    })

    return df

