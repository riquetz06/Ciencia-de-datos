import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# ===========================
# 1. Subir archivo (Colab) o usar local
# ===========================
try:
    from google.colab import files
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
except:
    filename = "VAR_Financieras.csv"  # Cambia si no estás en Colab

# ===========================
# 2. Leer CSV
# ===========================
df = pd.read_csv(filename, parse_dates=True, index_col=0)

print("=== Primeras filas del archivo ===")
print(df.head())

# ===========================
# 3. Limpiar columnas: eliminar comas de miles y convertir a float
# ===========================
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar columnas vacías
df = df.dropna(axis=1, how='all')

# Verificar que haya suficientes columnas numéricas
numeric_cols = df.select_dtypes(include='number').columns
if len(numeric_cols) < 2:
    raise ValueError("Se necesitan al menos 2 columnas numéricas para un VAR.")
df = df[numeric_cols]

# ===========================
# 4. Limpiar filas con NaN
# ===========================
df = df.dropna()
if df.empty:
    raise ValueError("El DataFrame quedó vacío después de limpiar NaN.")

print("\nFilas y columnas después de limpieza:", df.shape)

# ===========================
# 5. Detectar precios o rendimientos
# ===========================
if (df > 0).all().all() and (df.diff().abs().max().max() > 1):
    if df.shape[0] < 2:
        raise ValueError("No hay suficientes filas para calcular rendimientos.")
    print("\nSe detectaron PRECIOS. Calculando rendimientos...")
    df = (df / df.shift(1) - 1).dropna()
else:
    print("\nSe detectaron RENDIMIENTOS. Usando datos directamente.")

if df.empty:
    raise ValueError("No hay datos suficientes después de preparar rendimientos.")

print("\n=== Primeras filas de la base lista para VAR ===")
print(df.head())

# ===========================
# 6. Ajustar modelo VAR
# ===========================
model = VAR(df)
results = model.fit(maxlags=2)
print("\n=== Resumen del VAR ===")
print(results.summary())

# ===========================
# 7. Funciones Impulso-Respuesta (IRF)
# ===========================
irf = results.irf(10)  # horizonte de 10 pasos
print("\n=== Primeros valores de IRF (primeros 5 pasos) ===")
print(irf.irfs[:5])

# Graficar IRF
irf.plot(orth=False)
plt.suptitle("Funciones Impulso-Respuesta (IRF)", fontsize=14)
plt.show()

