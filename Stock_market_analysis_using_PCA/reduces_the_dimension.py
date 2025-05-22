# Python-код, який виконує симуляцію даних для 10 компаній, проводить нормалізацію, 
# зменшує розмірність за допомогою PCA та будує графік результату

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# 1. Генеруємо штучні дані: 500 днів, 10 компаній
np.random.seed(42)
n_days = 500
n_companies = 10

# Створення даних з 3 центрами (імітація груп схожих компаній)
data, labels = make_blobs(n_samples=n_days, centers=3, n_features=n_companies, cluster_std=2.5)

# 2. Нормалізація даних
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. PCA – зменшення до 2 компонент
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# 4. Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='Set1', s=30)
plt.title("PCA-візуалізація цін акцій 10 компаній (згенеровані дані)")
plt.xlabel("Перша головна компонента")
plt.ylabel("Друга головна компонента")
plt.grid(True)
plt.tight_layout()
plt.show()