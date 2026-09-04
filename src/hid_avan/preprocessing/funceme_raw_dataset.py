import numpy as np
import pandas as pd

def extract_arrays_from_daily_pr_dataset(funceme_txt_path: str,
                                         missing_pr_value: float | None = -999.) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Divide um dataset de séries de precipitações diárias da FUNCEME, estando em seu formato bruto, em 3 datasets em formato arrays,
    opcionalmente substituindo valores faltantes escritos como códigos numéricos (como -999) para NaN.

    Args:
        funceme_txt_path (str): Caminho do dataset bruto em .txt;
        missing_pr_value (float or None, optional): Valor numérico correspondente às precipitações faltantes. Se "None", não substitui.

    Returns:
        np.ndarray[d]: IDs das "d" estações de coleta das precipitações;
        np.ndarray[d, 2]: Latitude e longitude (colunas 0 e 1, respectivamente) das "d" estações (dispostas nas linhas);
        np.ndarray[n, d]: "d" séries de precipitações diárias, cada qual com "n" amostras.
    """
    complete_array = np.loadtxt(funceme_txt_path)

    id_array = np.int32( complete_array[:, 0] )

    coord_array = complete_array[:, [2, 1]]
    
    pr_array = complete_array[:, 3:].T
    if missing_pr_value is not None:
        pr_array = np.where(pr_array == missing_pr_value, np.nan, pr_array)

    return id_array, coord_array, pr_array

def build_timeseries_df_from_daily_pr_dataset(funceme_txt_path: str,
                                              date_start: str, date_end: str,
                                              missing_pr_value: float | None = -999.) -> pd.DataFrame:
    """
    Monta um dataframe de séries temporais a partir de um dataset de precipitações diárias da FUNCEME, estando em seu formato bruto,
    opcionalmente substituindo valores faltantes escritos como códigos numéricos (como -999) para NaN.

    Args:
        funceme_txt_path (str): Caminho do dataset bruto em .txt;
        date_start (str): Data do dia inicial no formato 'AAAA-MM-DD';
        date_end (str): Data do dia final no formato 'AAAA-MM-DD';
        missing_pr_value (float or None, optional): Valor numérico correspondente às precipitações faltantes. Se "None", não substitui.

    Returns:
        pd.DataFrame: Séries temporais dispostas nas colunas, identificadas pelo ID da estação, com a coluna 0 constando as datas.
    """
    id_array, _, pr_array = extract_arrays_from_daily_pr_dataset(funceme_txt_path=funceme_txt_path, missing_pr_value=missing_pr_value)
    n_stations = id_array.shape[0]
    
    df_pr = pd.DataFrame(pr_array, columns=id_array)
    df_pr['date'] = pd.date_range(start=date_start, end=date_end, freq='D')
    df_pr = df_pr.iloc[:, [-1] + list(range(n_stations))] 

    return df_pr