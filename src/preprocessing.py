# src/preprocessing.py
"""
Módulo de preprocesamiento de datos:
- Limpieza de datos
- Normalización de variables categóricas
- Discretización de variables continuas
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame eliminando valores infinitos y nulos.

    Parameters:
        df (pd.DataFrame): DataFrame original

    Returns:
        pd.DataFrame: DataFrame limpio
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def normalizar(df: pd.DataFrame, var: str, umbral: float = 0.05) -> tuple:
    """
    Normaliza una variable discreta agrupando categorías con baja frecuencia en 'Otros'.

    Parameters:
        df (pd.DataFrame): DataFrame con la variable
        var (str): Nombre de la columna a normalizar
        umbral (float): Frecuencia mínima para mantener categoría

    Returns:
        tuple: (nombre de variable, diccionario de mapeo)
    """
    aux = df[var].value_counts(normalize=True).to_frame(name='freq')
    aux['map'] = np.where(aux['freq'] < umbral, 'Otros', aux.index)
    if aux.loc[aux['map'] == 'Otros', 'freq'].sum() < umbral:
        aux['map'].replace({'Otros': aux.iloc[0]['map']}, inplace=True)
    return var, aux['map'].to_dict()

def discretizar_continuas(df: pd.DataFrame, varc: list, n_bins: int = 5) -> tuple:
    """
    Discretiza variables continuas usando quantiles.

    Parameters:
        df (pd.DataFrame): DataFrame con las variables
        varc (list): Lista de variables continuas
        n_bins (int): Número de bins

    Returns:
        tuple: (DataFrame con variables discretizadas, lista de nombres de variables discretizadas)
    """
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    kb.fit(df[varc])
    vardisc = [f'disc_{v}' for v in varc]
    df[vardisc] = kb.transform(df[varc]).astype(int)

    # Etiquetado de bins
    for v, d in zip(vardisc, map(lambda z: dict(enumerate([f'({t[0]}|{t[1]}]' for t in zip(map(str, z), map(str, z[1:]))])), kb.bin_edges_)):
        df[v] = df[v].replace(d)

    return df, vardisc
