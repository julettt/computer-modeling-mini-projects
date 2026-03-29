import numpy as np
import matplotlib.pyplot as plt

L = 25
k_rep = 50
k_attr = 1
dt = 0.01
alpha = 0.1
k_d0 = 0.1
k_a = 0.01

Cell_Processes_Check = 3
steps = 500

n_initial = 10



def closest_image(r1, r2, L):
    r12 = r2 - r1
    r12 -= L * np.round(r12 / L)
    return r12


class Cell:
    def __init__(self, r, R, daughter_of_id=None, is_mutated = False):
        self.r = np.array(r)
        self.R = R
        self.time_alive = 0
        self.daughter_of_id = daughter_of_id
        self.is_dividing = False
        self.is_mutated = is_mutated


cells = []
for i in range(n_initial):
    mutated = True if i == 0 else False
    cells.append(Cell(np.random.uniform(0, L, 2), np.random.normal(1, 0.1), is_mutated = mutated))

history_time = []
history_n_healthy = []
history_n_mutated = []

#grid to narrow neighbourhood
R_max = 1.5
grid_size = R_max * 1.1
n_grid = int(L / grid_size) + 1

for step in range(steps):

    grid = [[[] for _ in range(n_grid)] for _ in range(n_grid)]

    for idx, cell in enumerate(cells):
        ix = int(cell.r[0] // grid_size)
        iy = int(cell.r[1] // grid_size)
        grid[ix][iy].append(idx)

    N = len(cells)
    forces = np.zeros((N, 2)) #total forces acting on each cell
    neighbours = np.zeros(N)

    for idx, cell in enumerate(cells):
        ix = int(cell.r[0] // grid_size)
        iy = int(cell.r[1] // grid_size)

        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:

                nx = (ix + x) % n_grid
                ny = (iy + y) % n_grid

                for jdx in grid[nx][ny]:

                    if jdx <= idx:
                        continue
                    
                    other = cells[jdx]

                    r_vec = closest_image(cell.r, other.r, L)
                    dist = np.sqrt(np.dot(r_vec, r_vec))

                    sumR = cell.R + other.R
                    sumR_attr = 1.1 * sumR

                    if dist > sumR_attr:
                        continue

                    
                    r_hat = r_vec / dist if dist > 0 else np.zeros(2)

                    if dist <= sumR_attr:
                        neighbours[idx] += 1
                        neighbours[jdx] += 1

                    if dist < sumR:
                        f = -k_rep * (sumR - dist) * r_hat
                    elif dist <= sumR_attr:
                        f = k_attr * r_hat
                    else:
                        continue

                    
                    if (cell.daughter_of_id == id(other)) or (other.daughter_of_id == id(cell)):
                        f *= 0.5

                    forces[idx] += f
                    forces[jdx] -= f

 
    for i, cell in enumerate(cells):
        n = np.random.normal(0, 1, 2)
        cell.r += forces[i] * dt + np.sqrt(alpha * dt) * n
        cell.r %= L
        cell.time_alive += dt


    if step % Cell_Processes_Check == 0:

        to_die = []
        new_cells = []

        id_map = {id(c): c for c in cells}

        for i, cell in enumerate(cells):

            if cell.is_dividing:

                #partner = mother or daughter
                partner = id_map.get(cell.daughter_of_id, None)

                if partner is None:
                    cell.is_dividing = False
                    continue

                dist = np.linalg.norm(closest_image(cell.r, partner.r, L))

                if dist > 0.98 * (cell.R + partner.R):
                    cell.is_dividing = False
                    partner.is_dividing = False
                    cell.daughter_of_id = None
                    partner.daughter_of_id = None
                continue

            z_ng = neighbours[i]

            #mutated cells have 10x probability to divide
            current_k_d0 = k_d0 * 10 if cell.is_mutated else k_d0

            p_div = max(0, (current_k_d0 * (1 - z_ng / 6))) * cell.time_alive
            p_die = k_a * cell.time_alive

            r = np.random.rand()

            if r < p_die:
                to_die.append(i)

            elif r < p_die + p_div:

                cell.is_dividing = True
                daughter_r = (cell.r + 0.01 * np.random.rand(2)) % L
                daughter = Cell(daughter_r, np.random.normal(1, 0.1), is_mutated=cell.is_mutated)
                daughter.is_dividing = True
                daughter.daughter_of_id = id(cell)
                cell.daughter_of_id = id(daughter)

                cell.R = np.random.normal(1, 0.1)
                cell.time_alive = 0

                new_cells.append(daughter)

        cells = [c for idx, c in enumerate(cells) if idx not in to_die]
        cells.extend(new_cells)

        history_time.append(step * dt)
        
        n_mutated = sum(1 for c in cells if c.is_mutated)
        n_healthy = len(cells) - n_mutated
        history_n_mutated.append(n_mutated)
        history_n_healthy.append(n_healthy)

        

#plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

for cell in cells:
    color = 'blue' if cell.is_mutated else 'yellow'
    circle = plt.Circle(cell.r, cell.R, color=color, ec='black')
    ax1.add_patch(circle)

ax1.set_xlim(0, L)
ax1.set_ylim(0, L)
ax1.set_aspect('equal')
ax1.set_title(f"t = {step*dt:.2f}")

ax2.plot(history_time, history_n_healthy, color='orange', label='Healthy cells')
ax2.plot(history_time, history_n_mutated, color='blue', label='Mutated cells')
ax2.set_xlabel('time')
ax2.set_ylabel('N_cells')
ax2.legend()

plt.tight_layout()
plt.show()