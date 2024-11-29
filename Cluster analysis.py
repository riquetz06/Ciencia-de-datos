import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Crear un conjunto de datos aleatorio
np.random.seed(42)
data = np.random.rand(100, 2)  # 100 muestras, 2 características

# Crear un DataFrame
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

# Estandarizar los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)
df['Cluster'] = kmeans.labels_

# Crear una gráfica de los clústeres
plt.figure(figsize=(8, 6))
plt.scatter(df['Feature1'], df['Feature2'], c=df['Cluster'], cmap='viridis', marker='o')

# Añadir etiquetas y título
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Análisis de Clústeres (K-means)')
plt.grid(True)
plt.show()
