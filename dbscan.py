import math

import pygame
from matplotlib import pyplot as plt
import numpy as np

# Шевяков 11-502

dataset = np.empty((0, 2), dtype='f')


def create_data(position):
    (x, y) = position
    r = np.random.uniform(0, 30)
    phi = np.random.uniform(0, 2 * np.pi)
    coord = [x + r * np.cos(phi), y + r * np.sin(phi)]
    global dataset
    dataset = np.append(dataset, [coord], axis=0)


radius = 2
color = (0, 0, 255)
thickness = 0

bg_color = (255, 255, 255)
(width, heigth) = (640, 480)
screen = pygame.display.set_mode((width, heigth))
pygame.display.set_caption("DB_SCAN")
running = True
pushing = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pushing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            pushing = False

    if pushing and np.random.uniform() > .9:
        create_data(pygame.mouse.get_pos())

    screen.fill(bg_color)

    for i, data in enumerate(dataset):
        pygame.draw.circle(screen, color, (int(data[0]), int(data[1])), radius, thickness)

    pygame.display.flip()

pygame.quit()

colors = np.array(
    ['#377eb8', '#ff7f00', '#4daf4a', '#800000', '#ff0000', '#00ff00', '#00FFFF', '#000080', '#FAF0BE', '#CD00CD',
     '#F0F8FF'])


class DB_SCAN():
    def __init__(self, dataset, metric, eps=30, min_samples=2):
        self.dataset = dataset
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = 0
        self.clusters = {0: []}
        self.visited = set()
        self.clustered = set()
        self.fitted = False
        self.metric = metric
        self.fit()

    def get_dist(self, list1, list2):
        if self.metric == 'euc_dist':
            dist = math.sqrt(sum((i - j) ** 2 for i, j in zip(list1, list2)))
        elif self.metric == 'euc_dist2':
            dist = sum((i - j) ** 2 for i, j in zip(list1, list2))
        elif self.metric == 'city_block_dist':
            dist = sum(math.fabs(i - j) for i, j in zip(list1, list2))
        elif self.metric == 'cheb_dist':
            dist = max(math.fabs(i - j) for i, j in zip(list1, list2))
        return dist

    def get_region(self, data):
        return [list(q) for q in self.dataset if self.get_dist(data, q) < self.eps]

    def fit(self):
        for p in self.dataset:
            if tuple(p) in self.visited:
                continue
            self.visited.add(tuple(p))
            neighbours = self.get_region(p)
            if len(neighbours) < self.min_samples:
                self.clusters[0].append(list(p))
            else:
                self.n_clusters += 1
                self.expand_cluster(p, neighbours)
        self.fitted = True

    def expand_cluster(self, p, neighbours):
        if self.n_clusters not in self.clusters:
            self.clusters[self.n_clusters] = []
        self.clustered.add(tuple(p))
        self.clusters[self.n_clusters].append(list(p))
        while neighbours:
            q = neighbours.pop()
            if tuple(q) not in self.visited:
                self.visited.add(tuple(q))
                q_neighbours = self.get_region(q)
                if len(q_neighbours) > self.min_samples:
                    neighbours.extend(q_neighbours)
            if tuple(q) not in self.clustered:
                self.clustered.add(tuple(q))
                self.clusters[self.n_clusters].append(q)
                if q in self.clusters[0]:
                    self.clusters[0].remove(q)

    def get_labels(self):
        labels = np.array([])
        if not self.fitted:
            self.fit()
        for data in self.dataset:
            for i in range(self.n_clusters + 1):
                if list(data) in self.clusters[i]:
                    labels = np.append(labels, i).astype(int)
        return labels


metrics = ['euc_dist', 'euc_dist2', 'city_block_dist', 'cheb_dist']
test = DB_SCAN(dataset, metrics[3], 30, 2)
pred = test.get_labels()

plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[pred])
plt.show()
