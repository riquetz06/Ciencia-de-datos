import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Lista de precios de ORCL
precios = [
166.03,166.32,165.69,162.03,163.14,154.5,153.92,156.31,158.31,159.54,161.03,172.57,184.22,186.47,183.6,158.28,
164,162.02,170.38,170.06,168.6,167.89,171.66,172.35,174.46,178.92,177.19,172.22,173.86,174.16,179.8,181.52,176,
167.81,169.96,168.54,172.47,164.76,166.06,162.02,157.47,161.56,150.94,155.16,148.79,144.18,150.89,147.66,149.27,
154.01,149.45,152.45,152.72,152.23,154.87,153.93,147.8,145.78,140.87,139.81,141.94,145.86,137.23,128.27,127.16,
124.5,139.69,133.35,132.35,134.64,133.94,129.76,128.62,122.82,127.24,131.4,137.51,138.49,140.14,140.79,140.72,
145.49,150.73,149.29,147.7,149.37,150.3,150.34,157.22,162.27,162.95,159.4,160.49,159.64,160.31,157.18,157.31,
155.97,161.91,163.85,162.9,165.53,166.57,169.14,168.1,171.14,174.02,177.15,177.48,176.38,199.86,215.22,211.1,
208.18,210.87,205.17,207.04,215.27,210.72,212.82,210.24,218.63,218.96,229.98,237.32,232.26,234.5,235.81,235,
230.56,229.28,234.96,241.3,248.75,245.45,243.54,238.11,241.9,242.83,245.12,247.71,249.98,250.6,253.77,244.42,
252.53,255.67,256.43,249.39,250.05,252.68,253.86,244.18,244.96,248.09]

# Crear DataFrame con índice de fechas ficticias
fechas = pd.date_range(start='2025-01-01', periods=len(precios), freq='D')
df = pd.DataFrame({'ORCL': precios}, index=fechas)

# Ajustar modelo ARIMA (p,d,q) = (2,1,2) como punto de partida
modelo = ARIMA(df['ORCL'], order=(2,1,2))
resultado = modelo.fit()

# Mostrar resumen del modelo
print(resultado.summary())

# Predicción de los próximos 15 días
pred = resultado.forecast(steps=15)
fechas_pred = pd.date_range(start=fechas[-1] + pd.Timedelta(days=1), periods=15)
df_pred = pd.DataFrame({'Predicción': pred}, index=fechas_pred)

# Visualización
plt.figure(figsize=(12,6))
plt.plot(df['ORCL'], label='Precio ORCL')
plt.plot(df_pred['Predicción'], label='Predicción ARIMA', color='red')
plt.title('Modelo ARIMA para ORCL')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
# Convertir en DataFrame
df_pred = pd.DataFrame({'Predicción': pred}, index=fechas_pred)
print(df_pred)
