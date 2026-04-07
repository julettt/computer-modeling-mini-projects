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
t_max = 40
rb = 4
vp = v0 #predator velocity

#initial conditions
np.random.seed(120)
positions = np.random.uniform(low = 0, high = L, size = (N, 2))
directions = np.random.uniform(low = -np.pi, high = np.pi, size = N)

#bird of prey
pred_pos = np.random.uniform(low = 0, high = L, size = 2)
pred_dir = np.random.uniform(low = -np.pi, high = np.pi)

plt.figure(figsize = (6,6))

for step in range(int(t_max/dt)):

    birds_tree = cKDTree(positions, boxsize = [L,L])
    
    #predator impact
    dist_to_prey, index_of_prey = birds_tree.query(pred_pos)
    prey_pos = positions[index_of_prey]

    dx = prey_pos[0] - pred_pos[0]
    dy = prey_pos[1] - pred_pos[1]

    if dx > L/2:
        dx -= L
    if dx < -L/2:
        dx += L
    if dy > L/2:
        dy -= L
    if dy < -L/2:
        dy += L
    
    pred_dir = np.arctan2(dy, dx)
    
    dist = birds_tree.sparse_distance_matrix(birds_tree, max_distance = r, output_type = 'coo_matrix')

    data = np.exp(1j * directions[dist.col])
    orientations_matrix = sparse.coo_matrix((data, (dist.row, dist.col)), shape = (N, N))

    sum_angles = np.array(orientations_matrix.sum(axis = 1)).flatten()

    #updating angles
    noise = a * np.random.uniform(-np.pi, np.pi, size = N) 
    directions = np.angle(sum_angles) + noise

    scared_birds = birds_tree.query_ball_point(pred_pos, rb)

    for idx in scared_birds:
        dx_escape = positions[idx, 0] - pred_pos[0]
        dy_escape = positions[idx, 1] - pred_pos[1]

        directions[idx] = np.arctan2(dy_escape, dx_escape) + a * np.random.uniform(-np.pi, np.pi)


    #updating positions
    positions[:, 0] += v0 * np.cos(directions) * dt
    positions[:, 1] += v0 * np.sin(directions) * dt
    positions %= L

    pred_pos[0] += vp * np.cos(pred_dir) * dt
    pred_pos[1] += vp * np.sin(pred_dir) * dt
    pred_pos %= L

    if step % 5 == 0:
        plt.clf()
        plt.quiver(positions[:, 0], positions[:, 1], np.cos(directions), np.sin(directions), directions, cmap = 'hsv')
        plt.quiver(pred_pos[0], pred_pos[1], np.cos(pred_dir), np.sin(pred_dir), color = 'black')
        plt.xlim(0, L)
        plt.ylim(0, L)
        plt.title(f"with bird of pray")
        plt.pause(0.01)

plt.show()