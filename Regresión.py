import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson, jarque_bera
import matplotlib.pyplot as plt

# ===========================================================
# 1. Datos de ejemplo (sustituye con tus datos)
# ===========================================================
df = pd.DataFrame({
    "y": [10, 12, 13, 15, 16, 18, 19, 21, 22, 24],
    "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "x2": [4, 5, 5, 6, 6, 7, 7, 8, 9, 10]
})

# ===========================================================
# 2. Gráfica de dispersión inicial (Y vs X1)
# ===========================================================
plt.scatter(df["x1"], df["y"])
plt.xlabel("x1")
plt.ylabel("y")
plt.title("Dispersión entre y y x1")
plt.show()

# ===========================================================
# 3. Variables para regresión
# ===========================================================
X = df[["x1", "x2"]]   # variables independientes
y = df["y"]            # variable dependiente

# Agregar constante
X = sm.add_constant(X)

# ===========================================================
# 4. Ajustar modelo OLS
# ===========================================================
modelo = sm.OLS(y, X).fit()

# Imprimir resumen general
print(modelo.summary())

# ===========================================================
# 5. Prueba Durbin-Watson
# ===========================================================
dw = durbin_watson(modelo.resid)
print("\nDurbin-Watson:", round(dw, 4))

if dw < 1.5:
    print("→ Posible autocorrelación positiva")
elif dw > 2.5:
    print("→ Posible autocorrelación negativa")
else:
    print("→ No hay evidencia fuerte de autocorrelación")

# ===========================================================
# 6. Prueba Jarque-Bera
# ===========================================================
jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(modelo.resid)
print("\nJarque-Bera:", round(jb_stat,4))
print("p-value:", round(jb_pvalue,4))
print("Skewness:", round(skew,4))
print("Kurtosis:", round(kurtosis,4))

if jb_pvalue > 0.05:
    print("→ No se rechaza normalidad de residuos")
else:
    print("→ Se rechaza normalidad de residuos")

# ===========================================================
# 7. Gráfico de residuos (opcional)
# ===========================================================
plt.scatter(modelo.fittedvalues, modelo.resid)
plt.axhline(y=0, color='black')
plt.xlabel("Valores ajustados")
plt.ylabel("Residuos")
plt.title("Residuos vs Ajustados")
plt.show()
