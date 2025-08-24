# scorecard_project.py
"""
Script principal que ejecuta todo el flujo del scorecard.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_data, normalizar, discretizar_continuas
from src.woe_iv import calcular_woe, calcular_iv
from src.modeling import entrenar_logit

# ---- Crear carpetas de salida ----
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

# ---- Función para graficar ROC ----
def graficar_roc(y_true, y_prob, filename):
    plt.figure(figsize=(10,5))
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc_score:.4f})', color='blue')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)  # guarda directamente en la ruta completa
    plt.close()

# ---- Cargar datos ----
df = pd.read_csv('data/UCI_Credit_Card.csv')
df.rename(columns={'default.payment.next.month':'target'}, inplace=True)
df = clean_data(df)

# ---- Variables ----
vard = ['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
varc = ['LIMIT_BAL','AGE','BILL_AMT4','BILL_AMT5','BILL_AMT6',
        'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
um = ['ID']
vart = ['target']
df = df[um + vart + varc + vard]

# ---- Normalización ----
mapa_norm = [normalizar(df, v) for v in vard]
for v, mapa in mapa_norm:
    df[f'n_{v}'] = df[v].replace(mapa)
varn = [f'n_{v}' for v in vard]

# ---- Discretización ----
df, vardisc = discretizar_continuas(df, varc)

# ---- Dividir datos ----
train, valid = train_test_split(df, test_size=0.3, stratify=df['target'])
train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)

# ---- Calcular WoE e IV ----
mapa_woe = [calcular_woe(train, v) for v in varn + vardisc]

for v, mapa in mapa_woe:
    train[f'woe_{v}'] = train[v].replace(mapa)
    if v in valid.columns:
        valid[f'woe_{v}'] = valid[v].replace(mapa)

ivr = pd.DataFrame([calcular_iv(train, v) for v in varn + vardisc],
                   columns=['variable', 'iv']).sort_values('iv', ascending=False)
ivr.to_csv('outputs/ivr.csv', index=False)

# ---- Selección de variables ----
best = [f'woe_{v}' for v in ivr[ivr['iv'] > 0.01]['variable']]

# Comprobar columnas en valid
missing_cols = [c for c in best if c not in valid.columns]
if missing_cols:
    print("Faltan columnas en valid:", missing_cols)

# ---- Entrenar modelo ----
mod = entrenar_logit(train, best)

# ---- Probabilidades ----
y_train_prob = mod.predict_proba(train[best])[:,1]
y_valid_prob = mod.predict_proba(valid[best])[:,1]

# ---- Graficar ROC ----
graficar_roc(train['target'], y_train_prob, 'outputs/figures/roc_train.png')
graficar_roc(valid['target'], y_valid_prob, 'outputs/figures/roc_valid.png')

# ---- Calcular puntos de score ----
pdo = 12
base = 76
base_odds = 1
factor = pdo / np.log(2)
offset = base - factor * np.log(base_odds)

for v, beta in zip(best, mod.coef_[0]):
    train[f'p_{v}'] = (-train[v]*beta + mod.intercept_[0]/len(best))*factor + offset/len(best)

train['score'] = train[[f'p_{v}' for v in best]].sum(axis=1)
train['r_score'] = pd.cut(train['score'], bins=range(0, 120, 20), include_lowest=True).astype(str)

# ---- Pivot table para gráfico ----
piv = train.pivot_table(index='r_score', columns='target', values='ID', aggfunc='count', fill_value=0)
piv['total'] = piv.sum(axis=1)
piv['pct_default'] = piv[1] / piv['total']
piv['pct_non_default'] = piv[0] / piv['total']
piv_reset = piv.reset_index()

# ---- Gráfico de distribución de score ----
fig = px.bar(
    piv_reset,
    x='r_score',
    y=['pct_default', 'pct_non_default'],
    title='Probabilidad de incumplimiento según score',
    barmode='stack',
    color_discrete_map={'pct_default': 'gold', 'pct_non_default': 'lightblue'},
    labels={'r_score': 'Rango de score', 'value': 'Proporción de clientes', 'variable': 'Estado de pago'}
)
fig.show()


# Construir scorecard
scorecard = []
for v in best:
    nombre  = "_".join(v.split('_')[1:])
    aux = train[[nombre,f'p_{v}']].copy().drop_duplicates().sort_values(by=nombre)
    aux.columns = ['atributo','puntaje']
    aux.insert(0,'característica', "_".join(nombre.split('_')[1:]))
    scorecard.append(aux.reset_index(drop=True))

scorecard = pd.concat(scorecard, ignore_index=True)

# Guardar scorecard como CSV
scorecard.to_csv("outputs/scorecard.csv", index=False)

print("Scorecard guardado en outputs/scorecard.csv")