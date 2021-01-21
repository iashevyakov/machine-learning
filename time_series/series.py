import math
import numpy as np
import time
import urllib.request
import json
import matplotlib.pyplot as plt

# 28-й маршрут, остановка - ул. Гоголя
bus_stop_latitude = 55.795643
bus_stop_longtude = 49.132072

garagNumb = '16171'


def get_dist(list1, list2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(list1, list2)))


def find_dist(data):
    for d in data:
        if d['data']['GaragNumb'] == garagNumb:
            longitude = float(d['data']['Longitude'])
            latitude = float(d['data']['Latitude'])
            return get_dist([bus_stop_latitude, bus_stop_longtude], [latitude, longitude])


url = "http://data.kzn.ru:8082/api/v0/dynamic_datasets/bus.json"


# использовалось во время получения данных с сайта
# одно деление шкалы времени - 3 минуты
def fetch_data():
    count = 0
    while count <= 70:
        req = urllib.request.Request(url)
        r = urllib.request.urlopen(req).read()
        data = json.loads(r.decode('utf-8'))
        row = [count, find_dist(data)]
        with open('data.txt', 'a') as file:
            file.write(str(row[0]) + ' ' + str(row[1]) + '\n')
        time.sleep(180)
        count += 1


def dexp_smoothing(series, alpha, beta):
    results = []
    leveles = []
    trends = []
    for n in range(1, len(series)):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        else:
            last_level, level = level, alpha * series[n] + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
        results.append(level + trend)
        leveles.append(level)
        trends.append(trend)
    return results, leveles, trends


dataset = []
f = open('data.txt', 'r')
for line in f.readlines():
    time = int(line[:-1].split(' ')[0])
    dist = float(line[:-1].split(' ')[1])
    dataset.append([time, dist])
f.close()
dataset = np.array(dataset)

print(dataset)

x, y = dataset[:, 0], dataset[:, 1]

plt.figure(figsize=(20, 10))
plt.scatter(x, y)
y_r, y_l, y_t = dexp_smoothing(y, .5, .1)
plt.plot(x[:len(y_r)], y_r, c='red', linewidth=5, label='result')
plt.plot(x[:len(y_r)], y_l, c='green', linewidth=5, label='level')
plt.plot(x[:len(y_r)], y_t, c='blue', linewidth=5, label='trend')
plt.legend(loc='best')
plt.show()
