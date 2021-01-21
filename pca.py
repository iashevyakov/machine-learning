import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

num = 300
dataset = np.array([[i, 2 * i + np.random.uniform(-30, 30), np.random.uniform(-100, 100)] for i in range(num)])

dataset_cntr = dataset - dataset.mean(axis=0)

fig = plt.figure(1, figsize=(6, 5))

ax = Axes3D(fig, elev=48, azim=134)

ax.scatter(dataset_cntr[:, 0], dataset_cntr[:, 1], dataset_cntr[:, 2], c='black', cmap=plt.cm.nipy_spectral)

plt.show()

covmat = np.cov(dataset_cntr, rowvar=False)

vals, vects = np.linalg.eig(covmat)
print("Собственные значения и вектора", vals, vects, sep='\n')

# находим индексы двух максимальных собственных значений
indexes_of_max = np.argpartition(vals, -2)[-2:]
indexes_of_max = np.sort(indexes_of_max)

print("\nИндексы макс.собств.зн-й", indexes_of_max, sep=':')

vect1 = vects[indexes_of_max[0]].reshape(3, -1)
vect2 = vects[indexes_of_max[1]].reshape(3, -1)

print("\nНовый базис", vect1, vect2, sep='\n')

coord1 = np.dot(dataset_cntr, vect1)
coord2 = np.dot(dataset_cntr, vect2)

coords = []
for i, j in zip(coord1, coord2):
    coords.append([i[0], j[0]])
coords = np.array(coords)

plt.figure()
plt.scatter(coords[:, 0], coords[:, 1], c='black')
plt.show()
