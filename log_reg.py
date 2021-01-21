import math
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def predict(teta, x):
    if sigmoid(np.dot(teta, x)) >= 0.5:
        return 1
    else:
        return 0


def log_reg(X, y):
    teta = np.zeros(len(X[0]))
    alpha = 0.01
    epochs = 1000
    for epoch in range(1, epochs):
        for i, x in enumerate(X):
            teta += alpha * (y[i] - sigmoid(np.dot(teta, x))) * x
    return teta


# dataset с 2-мя классами
breast_cancer = load_breast_cancer()
X_full = breast_cancer.data

X_cut = []
for x in X_full:
    X_cut.append(x[:3])
X_cut = np.array(X_cut)

y = breast_cancer.target

teta = log_reg(X_cut[:450], y[:450])
print('Веса признаков: ', teta)

pred = []

for i, x in enumerate(X_cut[450:]):
    pred.append(predict(teta, x))

accuracy = accuracy_score(y[450:], pred)
print('Точность предсказаний: ', accuracy)
