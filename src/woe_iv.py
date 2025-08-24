# src/woe_iv.py
"""
Módulo para calcular WoE e IV de variables.
"""

import pandas as pd
import numpy as np

def calcular_woe(df: pd.DataFrame, v: str, target: str = 'target') -> tuple:
    """
    Calcula el Weight of Evidence (WoE) para una variable.

    Parameters:
        df (pd.DataFrame): DataFrame
        v (str): Variable
        target (str): Variable objetivo

    Returns:
        tuple: (nombre variable, diccionario de WoE)
    """
    aux = df[[v, target]].assign(n=1)
    piv = aux.pivot_table(index=v, columns=target, values='n', aggfunc='sum', fill_value=0)
    piv /= piv.sum()
    piv['woe'] = np.log(piv[0] / piv[1])
    return v, piv['woe'].to_dict()

def calcular_iv(df: pd.DataFrame, v: str, target: str = 'target') -> tuple:
    """
    Calcula el Information Value (IV) para una variable.

    Parameters:
        df (pd.DataFrame): DataFrame
        v (str): Variable
        target (str): Variable objetivo

    Returns:
        tuple: (nombre variable, IV)
    """
    aux = df[[v, target]].assign(n=1)
    piv = aux.pivot_table(index=v, columns=target, values='n', aggfunc='sum', fill_value=0)
    piv /= piv.sum()
    piv['woe'] = np.log(piv[0]/piv[1])
    piv['iv'] = (piv[0]-piv[1])*piv['woe']
    return v, piv['iv'].sum()
