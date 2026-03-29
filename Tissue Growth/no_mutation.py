import numpy as np
import matplotlib.pyplot as plt

L = 25
k_rep = 50
k_attr = 1
dt = 0.01
alpha = 0.1
k_d0 = 0.1
k_a = 0.001

Cell_Processes_Check = 3
steps = 1000

n_initial = 10


def closest_image(r1, r2, L):
    r12 = r2 - r1
    r12 -= L * np.round(r12 / L)
    return r12


class Cell:
    def __init__(self, r, R, daughter_of_id=None):
        self.r = np.array(r)
        self.R = R
        self.time_alive = 0
        self.daughter_of_id = daughter_of_id
        self.is_dividing = False



cells = [Cell(np.random.uniform(0, L, 2), np.random.normal(1, 0.1)) for _ in range(n_initial)]

history_time = []
history_n_cells = []

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

            p_div = max(0, (k_d0 * (1 - z_ng / 6))) * cell.time_alive
            p_die = k_a * cell.time_alive

            r = np.random.rand()

            if r < p_die:
                to_die.append(i)

            elif r < p_die + p_div:

                cell.is_dividing = True
                daughter_r = (cell.r + 0.01 * np.random.rand(2)) % L
                daughter = Cell(daughter_r, np.random.normal(1, 0.1))
                daughter.is_dividing = True
                daughter.daughter_of_id = id(cell)
                cell.daughter_of_id = id(daughter)

                cell.R = np.random.normal(1, 0.1)
                cell.time_alive = 0

                new_cells.append(daughter)

        cells = [c for idx, c in enumerate(cells) if idx not in to_die]
        cells.extend(new_cells)

        history_time.append(step * dt)
        history_n_cells.append(len(cells))
        

#plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

for cell in cells:
    circle = plt.Circle(cell.r, cell.R, color='yellow', ec='black')
    ax1.add_patch(circle)

ax1.set_xlim(0, L)
ax1.set_ylim(0, L)
ax1.set_aspect('equal')
ax1.set_title(f"t = {step*dt}")

ax2.plot(history_time, history_n_cells)
ax2.set_xlabel('time')
ax2.set_ylabel('N_cells')

plt.tight_layout()
plt.show()