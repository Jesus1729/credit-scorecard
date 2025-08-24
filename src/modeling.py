# src/modeling.py
"""
Módulo para entrenar el modelo de scorecard, generar predicciones y calcular el score.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os

def entrenar_logit(df_train: pd.DataFrame, best_vars: list, target: str = 'target') -> LogisticRegression:
    """
    Entrena un modelo de regresión logística.

    Parameters:
        df_train (pd.DataFrame): DataFrame de entrenamiento
        best_vars (list): Variables seleccionadas para el modelo
        target (str): Variable objetivo

    Returns:
        LogisticRegression: Modelo entrenado
    """
    mod = LogisticRegression(class_weight='balanced')
    mod.fit(df_train[best_vars], df_train[target])
    return mod

def aplicar_modelo(df: pd.DataFrame, mod, mapa_woe: list):
    """
    Aplica las transformaciones WoE y predicciones al DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame
        mod: Modelo entrenado
        mapa_woe (list): Lista de tuplas (variable, diccionario WoE)

    Returns:
        pd.DataFrame: DataFrame con probabilidades
    """
    for v, mapa in mapa_woe:
        df[f'woe_{v}'] = df[v].replace(mapa)
    df['prob'] = mod.predict_proba(df[[f'woe_{v}' for v, _ in mapa_woe]])[:,1]
    return df

def graficar_roc(y_true, y_prob, nombre_archivo='roc_curve.png'):
    """
    Genera curva ROC y guarda el gráfico.

    Parameters:
        y_true (array): Valores reales
        y_prob (array): Probabilidades predichas
        nombre_archivo (str): Nombre del archivo PNG
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8,5))
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.4f})', color='purple')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True)
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig(f'outputs/figures/{nombre_archivo}', dpi=300, bbox_inches='tight')
    plt.show()
