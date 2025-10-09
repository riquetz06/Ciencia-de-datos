"""
GLM ASEGURADORA - Versión (a) con datos simulados
-------------------------------------------------
Este script muestra cómo aplicar distintos Modelos Lineales Generalizados (GLM)
en el sector asegurador, usando datos simulados realistas.

Modelos incluidos:
1. Gaussiano → Monto promedio del siniestro
2. Logístico → Probabilidad de siniestro
3. Poisson → Frecuencia de siniestros
4. Gamma → Severidad del siniestro
5. Binomial Negativo → Frecuencia con sobredispersión

Incluye:
- Explicaciones detalladas (didácticas)
- Gráficos de dispersión y líneas ajustadas
- Pruebas de homocedasticidad, autocorrelación y normalidad
"""

# ======================
# 1. Importar librerías
# ======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera, kstest

sns.set(style="whitegrid", context="talk")

# Crear carpeta para guardar los gráficos
import os
os.makedirs("resultados", exist_ok=True)

# ======================
# 2. Modelo GAUSSIANO
# ======================
print("\n📘 MODELO 1: GAUSSIANO (Monto del siniestro ~ Edad)\n")

np.random.seed(1)
n = 300
edad = np.random.randint(18, 80, n)
monto = 5000 + 200 * edad + np.random.normal(0, 5000, n)

df = pd.DataFrame({"Edad": edad, "Monto": monto})

X = sm.add_constant(df["Edad"])
modelo_gauss = sm.GLM(df["Monto"], X, family=sm.families.Gaussian()).fit()
print(modelo_gauss.summary())

# Gráfico: Dispersión y línea ajustada
sns.scatterplot(x="Edad", y="Monto", data=df, alpha=0.6)
edad_seq = np.linspace(df["Edad"].min(), df["Edad"].max(), 100)
pred = modelo_gauss.predict(sm.add_constant(edad_seq))
plt.plot(edad_seq, pred, color="red", label="Ajuste GLM")
plt.title("Monto del siniestro vs Edad (Gaussiano)")
plt.legend()
plt.savefig("resultados/gaussiano.png", dpi=150)
plt.show()

# Diagnósticos
resid = modelo_gauss.resid_response
fitted = modelo_gauss.fittedvalues

# Homocedasticidad
bp_test = het_breuschpagan(resid, X)
print("Breusch-Pagan:", bp_test)

# Autocorrelación
dw = durbin_watson(resid)
print("Durbin-Watson:", dw)

# Normalidad
jb = jarque_bera(resid)
ks = kstest((resid - resid.mean())/resid.std(), "norm")
print("Jarque-Bera:", jb)
print("Kolmogorov-Smirnov:", ks)

sns.residplot(x=fitted, y=resid, lowess=True, line_kws={"color": "red"})
plt.title("Residuos vs Ajustados (Gaussiano)")
plt.savefig("resultados/residuos_gaussiano.png", dpi=150)
plt.show()

# ======================
# 3. Modelo LOGÍSTICO
# ======================
print("\n📘 MODELO 2: LOGÍSTICO (Probabilidad de siniestro)\n")

np.random.seed(2)
n = 400
edad = np.random.randint(18, 80, n)
ingreso = np.random.normal(30000, 8000, n)
# Probabilidad real simulada
prob_siniestro = 1 / (1 + np.exp(-(-5 + 0.05 * edad - 0.0001 * ingreso)))
siniestro = np.random.binomial(1, prob_siniestro)

df2 = pd.DataFrame({"Edad": edad, "Ingreso": ingreso, "Siniestro": siniestro})

X = sm.add_constant(df2[["Edad", "Ingreso"]])
modelo_logit = sm.GLM(df2["Siniestro"], X, family=sm.families.Binomial()).fit()
print(modelo_logit.summary())

sns.scatterplot(x="Edad", y="Siniestro", data=df2, alpha=0.5)
edad_seq = np.linspace(df2["Edad"].min(), df2["Edad"].max(), 100)
pred = modelo_logit.predict(sm.add_constant(pd.DataFrame({"Edad": edad_seq, "Ingreso": [30000]*100})))
plt.plot(edad_seq, pred, color="red")
plt.title("Probabilidad de siniestro vs Edad (Logístico)")
plt.savefig("resultados/logit.png", dpi=150)
plt.show()

# ======================
# 4. Modelo POISSON
# ======================
print("\n📘 MODELO 3: POISSON (Frecuencia de siniestros)\n")

np.random.seed(3)
n = 300
exposicion = np.random.uniform(1, 5, n)
edad = np.random.randint(18, 80, n)
frecuencia = np.random.poisson(np.exp(-1 + 0.02 * edad), n)

df3 = pd.DataFrame({"Edad": edad, "Frecuencia": frecuencia, "Exposicion": exposicion})

X = sm.add_constant(df3["Edad"])
modelo_pois = sm.GLM(df3["Frecuencia"], X, family=sm.families.Poisson()).fit()
print(modelo_pois.summary())

sns.scatterplot(x="Edad", y="Frecuencia", data=df3)
plt.plot(df3["Edad"], modelo_pois.fittedvalues, "r.", label="Ajuste")
plt.title("Frecuencia vs Edad (Poisson)")
plt.legend()
plt.savefig("resultados/poisson.png", dpi=150)
plt.show()

# ======================
# 5. Modelo GAMMA
# ======================
print("\n📘 MODELO 4: GAMMA (Severidad de siniestros)\n")

np.random.seed(4)
n = 300
edad = np.random.randint(18, 80, n)
severidad = np.random.gamma(shape=2, scale=2000 + 40 * edad)

df4 = pd.DataFrame({"Edad": edad, "Severidad": severidad})

X = sm.add_constant(df4["Edad"])
modelo_gamma = sm.GLM(df4["Severidad"], X, family=sm.families.Gamma(link=sm.families.links.log())).fit()
print(modelo_gamma.summary())

sns.scatterplot(x="Edad", y="Severidad", data=df4, alpha=0.6)
edad_seq = np.linspace(df4["Edad"].min(), df4["Edad"].max(), 100)
pred = modelo_gamma.predict(sm.add_constant(edad_seq))
plt.plot(edad_seq, pred, color="red")
plt.title("Severidad vs Edad (Gamma)")
plt.savefig("resultados/gamma.png", dpi=150)
plt.show()

# ======================
# 6. Modelo BINOMIAL NEGATIVO
# ======================
print("\n📘 MODELO 5: BINOMIAL NEGATIVO (Frecuencia con sobredispersión)\n")

np.random.seed(5)
n = 300
edad = np.random.randint(18, 80, n)
# Simulación con sobredispersión
frecuencia = np.random.negative_binomial(2, 1/(1 + np.exp(-(-2 + 0.03 * edad))))

df5 = pd.DataFrame({"Edad": edad, "Frecuencia": frecuencia})

X = sm.add_constant(df5["Edad"])
modelo_nb = sm.GLM(df5["Frecuencia"], X, family=sm.families.NegativeBinomial()).fit()
print(modelo_nb.summary())

sns.scatterplot(x="Edad", y="Frecuencia", data=df5)
plt.plot(df5["Edad"], modelo_nb.fittedvalues, "r.", label="Ajuste")
plt.title("Frecuencia vs Edad (Binomial Negativo)")
plt.legend()
plt.savefig("resultados/neg_binomial.png", dpi=150)
plt.show()

print("\n✅ Todos los modelos GLM se han ejecutado con éxito y los gráficos están en la carpeta 'resultados/'.")
