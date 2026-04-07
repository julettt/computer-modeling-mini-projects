import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


npoints = 60
alpha = 1
beta = 1
rho = 0.5
Q = 1

num_iter = 50
num_ants = 20
start = 0
goal = 17

targets = [start, 1, 12, 32, goal]

max_steps = 100 #max steps given to ant for it to find their path

#generating graph
np.random.seed(111) #change to get random one
points = np.random.randint(0,15, size=(npoints, 2))
points = np.unique(points, axis=0)

max_iter = 100
itera = 0
while points.shape[0]-npoints<0:
    next_points = np.random.randint(0,10, size=(npoints-points.shape[0], 2))
    points = np.append(points, next_points, axis=0)
    points = np.unique(points, axis=0)
    itera += 1
    if itera>max_iter:
        break
npoints = points.shape[0]

D = Delaunay(points)
simplices = D.simplices

#adjacency matrix
adj = np.zeros((npoints, npoints))

for triplet in simplices:
    for i in range(3):
        for j in range(3):
            if i != j:
                adj[triplet[i]][triplet[j]] = 1


#weight matrix
weights = np.zeros((npoints, npoints))

#pheromone matrix
tau = np.zeros((npoints, npoints))

for i in range(npoints):
    for j in range(npoints):
        if adj[i][j] == 1:
            weights[i][j] = np.linalg.norm(points[i] - points[j])
            tau[i][j] = 1


#path calculation and simplification
def calculate_path(path):
    dist = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        dist += weights[u][v]
    return dist

def simplify_path(path):
    new_path = []
    for i in path:
        if i not in new_path:
            new_path.append(i)
        else:
            index = new_path.index(i)
            new_path = new_path[:index + 1]
    return new_path

#moving decision
def move_ant(current_pos):
    neighbours = [i for i in range(npoints) if adj[current_pos][i] == 1]

    values = []
    for n in neighbours:
        val = ((1/weights[current_pos][n]) **alpha) * (tau[current_pos][n]**beta)
        values.append(val)

    total_value = sum(values)
    probabilities = [v/total_value for v in values]

    return np.random.choice(neighbours, p = probabilities)

for _ in range(num_iter):
    all_paths = []

    #ants movements
    for ant in range(num_ants):
        current = start
        final_path = []
        success = True

        for i in range(len(targets) - 1):
            local_goal = targets[i + 1]
            segment_path = [current]
            steps = 0
   
            while current != local_goal and steps < max_steps:
                current = move_ant(current)
                segment_path.append(current)
                steps += 1

            if current != local_goal:
                success = False
                break
            
            else:
                simplified_segment = simplify_path(segment_path)

                if len(final_path) == 0:
                    final_path.extend(simplified_segment)

                else:
                    final_path.extend(simplified_segment[1:])

        if success:
            all_paths.append(final_path)

    #pheromones changes:
    #evaporation
    tau *= (1 - rho)

    #new pheromones
    for path in all_paths:
        L = calculate_path(path)

        if L > 0:
            delta_tau = Q / L

            for k in range(len(path) - 1):
                u = path[k]
                v = path[k + 1]
                
                tau[u][v] += delta_tau
                tau[v][u] += delta_tau


best_path = min(all_paths, key = calculate_path)
path_coordinates = points[best_path]
path_x = path_coordinates[:, 0]
path_y = path_coordinates[:, 1]

print(best_path)

#plotting
plt.figure(figsize = (9, 5))
plt.subplot(1, 2, 1)
plt.triplot(points[:, 0], points[:, 1], simplices, alpha = 0.6)
plt.scatter(points[:, 0], points[:, 1], color='C0')
for i, (x, y) in enumerate(points):
    plt.text(x + 0.2, y + 0.2, str(i), fontsize = 8, fontweight = 'bold')


plt.subplot(1, 2, 2)
plt.triplot(points[:, 0], points[:, 1], simplices, alpha = 0.6)
plt.scatter(points[:, 0], points[:, 1], color='C0')
plt.plot(path_x, path_y, color='red', linewidth=3, label='Shortest path')
plt.legend()


plt.tight_layout()
plt.show()