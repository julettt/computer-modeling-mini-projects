import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import scipy.sparse as sparse

L = 32
N = 5000
r = 1
v0 = 2
a = 0.15
dt = 0.1
t_max = 50

#initial conditions
np.random.seed(111)
positions = np.random.uniform(low = 0, high = L, size = (N, 2))
directions = np.random.uniform(low = -np.pi, high = np.pi, size = N)

plt.figure(figsize = (6,6))

for step in range(int(t_max/dt)):

    birds_tree = cKDTree(positions, boxsize = [L,L])
    dist = birds_tree.sparse_distance_matrix(birds_tree, max_distance = r, output_type = 'coo_matrix')

    data = np.exp(1j * directions[dist.col])
    orientations_matrix = sparse.coo_matrix((data, (dist.row, dist.col)), shape = (N, N))

    sum_angles = np.array(orientations_matrix.sum(axis = 1)).flatten()

    #updating angles
    noise = a * np.random.uniform(-np.pi, np.pi, size = N) 
    directions = np.angle(sum_angles) + noise

    #updating positions
    positions[:, 0] += v0 * np.cos(directions) * dt
    positions[:, 1] += v0 * np.sin(directions) * dt
    positions %= L

    if step % 5 == 0:
        plt.clf()
        plt.quiver(positions[:, 0], positions[:, 1], np.cos(directions), np.sin(directions), directions, cmap = 'hsv')
        plt.xlim(0, L)
        plt.ylim(0, L)
        plt.title(f"without bird of pray")
        plt.pause(0.01)

plt.show()