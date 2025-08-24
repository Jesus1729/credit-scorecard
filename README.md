# Credit Scorecard Project

Este proyecto construye un **scorecard de crédito** utilizando el dataset de pagos de tarjetas de crédito de [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset). El scorecard permite predecir la probabilidad de incumplimiento de los clientes y genera métricas clave para evaluar el desempeño del modelo.

---

## Estructura del proyecto

scorecard_proyecto/
│
├─ data/
│ └─ UCI_Credit_Card.csv # Dataset original
│
├─ src/
│ ├─ preprocessing.py # Funciones de limpieza, normalización y discretización
│ ├─ woe_iv.py # Funciones para cálculo de WoE e IV
│ └─ modeling.py # Funciones para entrenamiento y evaluación del modelo
│
├─ outputs/
│ ├─ figures/ # Gráficos generados (ROC, distribución de scores)
│ └─ ivr.csv (Tabla generada IVR)
| └─ scorecard.csv (Tabla generada scorecard)
│
├─ scorecard_project.py # Script principal
└─ README.md 

## Uso

Ejecuta el script principal:

´´´
python scorecard_project.py
´´´

Esto realizará:

- Limpieza y preprocesamiento de los datos.

- Cálculo de WoE (Weight of Evidence) e IV (Information Value) para las variables.

- Entrenamiento de un modelo de regresión logística balanceado.

- Generación de las curvas ROC (train y validación) y guardado en outputs/figures.

- Creación del scorecard final y guardado en outputs/scorecard.csv.

- Guardado de la tabla de IV en outputs/ivr.csv.

