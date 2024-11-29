import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Crear un conjunto de datos aleatorio
np.random.seed(42)
data = np.random.rand(100, 5)  # 100 muestras, 5 características
labels = np.random.choice([0, 1], size=100)  # Etiquetas binarias

# Crear un DataFrame
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
df['Label'] = labels
# Dividir el conjunto de datos
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dividir el conjunto de datos
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo LDA
lda = LinearDiscriminantAnalysis()

# Entrenar el modelo
lda.fit(X_train, y_train)

# Realizar predicciones
y_pred = lda.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy}')

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:')
print(conf_matrix)

# Informe de clasificación
class_report = classification_report(y_test, y_pred)
print('Informe de clasificación:')
print(class_report)

# Graficar los datos y las predicciones
plt.figure(figsize=(8, 6))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, marker='o', label='Actual')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, marker='x', label='Predicho')

# Añadir etiquetas y título
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Análisis Discriminante Lineal (LDA)')
plt.legend()
plt.show()
