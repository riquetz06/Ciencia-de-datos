import numpy as np
import pandas as pd
import os

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Crear carpeta para resultados
os.makedirs("resultados", exist_ok=True)

# ---------------------------
# Simulación de datos
# ---------------------------
np.random.seed(2025)
n = 400

edad = np.random.randint(18, 65, size=n)
ingreso = np.random.normal(50000, 15000, size=n).clip(10000, 150000)
educacion = np.random.randint(0, 3, size=n)  # 0: secundaria, 1: preparatoria, 2: universidad
num_hijos = np.random.randint(0, 5, size=n)

# Gaussiano: Consumo anual
consumo = 5000 + 0.6*ingreso + 200*edad + np.random.normal(0, 5000, n)

# Logístico: Probabilidad de desempleo
logit_prob = 1 / (1 + np.exp(-(1 - 0.05*edad - 0.5*educacion)))
desempleo = np.random.binomial(1, logit_prob)

# Poisson: Número de transacciones financieras
transacciones = np.random.poisson(lam=np.exp(-1 + 0.01*edad + 0.00001*ingreso), size=n)

# Gamma: Gasto en educación
gasto_educacion = np.random.gamma(shape=2, scale=1000 + 20*ingreso/1000 + 50*num_hijos)

# Binomial Negativo: Número de préstamos solicitados
prestamos = np.random.negative_binomial(2, 1/(1 + np.exp(-(-2 + 0.03*edad + 0.00002*ingreso))))

# ---------------------------
# Crear DataFrame
# ---------------------------
df_economia = pd.DataFrame({
    "Edad": edad,
    "Ingreso": ingreso,
    "Educacion": educacion,
    "Num_hijos": num_hijos,
    "Consumo": consumo,
    "Desempleo": desempleo,
    "Transacciones": transacciones,
    "Gasto_educacion": gasto_educacion,
    "Prestamos": prestamos
})

# Guardar CSV descargable
csv_path = "/mnt/data/datos_economia_ejemplo.csv"
df_economia.to_csv(csv_path, index=False)
csv_path
