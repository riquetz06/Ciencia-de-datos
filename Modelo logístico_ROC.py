import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# 1. Simulación de un dataset de Crédito
np.random.seed(42)
n_samples = 2000

edad = np.random.randint(18, 70, n_samples)
ingresos_anuales = np.random.exponential(scale=45000, size=n_samples) + 15000
ratio_deuda = np.random.uniform(0.05, 0.85, n_samples)
historial_impagos = np.random.choice([0, 1, 2], size=n_samples, p=[0.8, 0.15, 0.05])

# Generar la variable objetivo (default: 1 = impago, 0 = al corriente) con cierta lógica
scoring_prob = (0.05 * (60 - edad) + 0.1 * ratio_deuda * 100 + 1.5 * historial_impagos - 0.00002 * ingresos_anuales)
prob = 1 / (1 + np.exp(-scoring_prob))
# Agregar ruido y definir umbral para simular desbalance (aprox 15% de defaults)
default = (prob > np.percentile(prob, 85)).astype(int)

df_credito = pd.DataFrame({
    'edad': edad,
    'ingresos': ingresos_anuales,
    'ratio_deuda': ratio_deuda,
    'historial_impagos': historial_impagos,
    'default': default
})

display(df_credito.head())
print("Distribución de la variable objetivo (default):")
print(df_credito['default'].value_counts(normalize=True))

X = df_credito.drop(columns=['default'])
y = df_credito['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Obtener coeficientes para analizar el impacto de cada variable
coef_df = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': model.coef_[0]
}).sort_values(by='Coeficiente', ascending=False)

display(coef_df)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Calcular ROC y Gini
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
gini = 2 * roc_auc - 1

# Graficar Curva ROC
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title(f'Curva ROC - Gini Coeficiente: {gini:.2f}')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
