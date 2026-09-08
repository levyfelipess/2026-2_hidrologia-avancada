import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_basin_and_stations(df_basin: pd.DataFrame,
                            df_stations: pd.DataFrame,
                            factor_borders: float = 0.3,
                            plot_text_col: str | None = None,
                            special_points: dict | None = None) -> ("plt.Figure", "plt.Axes"):
    """
    Constrói e retorna os objetos de plotagem do contorno da bacia hidrográfica juntamente a estações pontuais.

    Args:
        df_basin (pd.DataFrame[n1, 2]): Dataframe com n1 pontos do contorno da bacia e colunas nomeadas como 'Latitude' e 'Longitude';
        
        df_stations (pd.DataFrame[n2, 2+d]): Dataframe com n2 estações e, no mínimo as duas colunas de coordenadas, nomeadas como
                'Latitude' e 'Longitude';
                
        factor_borders (float, optional): Fator que delimitará os limites de plotagem, calculado para ser uma extensão dos limites da bacia
                acrescidos ou diminuídos do fator x o comprimento da bacia naquela direção;
                
        plot_text_col (str or None, optional): Coluna de plotagem dos textos nos pontos das estações (por exemplo, o nome ou código). Se None,
                não plota nada.
                
        special_points (dict['df':pd.DataFrame[n3, 2], 'label':str or None] or None, optional): Dicionário contendo as informações de plotagem
                de pontos especiais: a chave 'df' remete ao dataframe com n3 pontos e as colunas de coordenadas nomeadas 'Latitude' e 'Longitude',
                e a chave 'label' remete à descrição que ficará na legenda, caso seja necessário (None para não plotar o label).

    Returns:
        plt.Figure, plt.Axes: Objetos de plotagem do matplotlib.
    """
    lat_min = df_basin['Latitude'].min() - np.ptp(df_basin['Latitude']) * factor_borders
    lat_max = df_basin['Latitude'].max() + np.ptp(df_basin['Latitude']) * factor_borders
    lon_min = df_basin['Longitude'].min() - np.ptp(df_basin['Longitude']) * factor_borders
    lon_max = df_basin['Longitude'].max() + np.ptp(df_basin['Longitude']) * factor_borders
    
    df_queried = df_stations.query(f'Latitude > {lat_min} and Latitude < {lat_max} and Longitude > {lon_min} and Longitude < {lon_max}')
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), layout='constrained')
    sns.scatterplot(ax=ax, data=df_basin, x='Latitude', y='Longitude', s=5, label='Contorno da Bacia')
    sns.scatterplot(ax=ax, data=df_queried, x='Latitude', y='Longitude', color='red', marker='x', s=50, label='Estações')
    if special_points is not None:
        sns.scatterplot(ax=ax, data=special_points['df'], x='Latitude', y='Longitude', color='black', marker='X', s=60, label=special_points['label'])
    if plot_text_col is not None:
        for lat, lon, name in zip(df_queried['Latitude'], df_queried['Longitude'], df_queried[plot_text_col]):
            ax.text(x=lat, y=lon, s=name, size=7, zorder=3)
    ax.set_xlim(lat_min, lat_max)
    ax.set_ylim(lon_min, lon_max)
    ax.grid(lw=.5)
    ax.legend()
    return fig, ax