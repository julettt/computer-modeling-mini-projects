import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.ndimage as nd

chromosome_size = 512
population_size = 15
world_size = 50
steps = 100
n_test = 5

rep_chance = 0.7
mutation_chance = 0.01
generations = 150

np.random.seed(1450)

#functions

def get_chromosome(chromosome_size):
    return np.random.randint(0, 2, chromosome_size)

'''
def encoding_old(grid):
    #axis = 0 -> przesunięcie góra (-1) /dół (1)
    #axis = 1 -> przesunięcie lewo (-1) /prawo (1)
    M0 = np.roll(np.roll(grid, 1, axis = 1), 1, axis = 0)
    M1 = np.roll(grid, 1, axis = 0)
    M2 = np.roll(np.roll(grid, -1, axis = 1), 1, axis = 0)
    M3 = np.roll(grid, 1, axis = 1)
    #M4 = grid
    M5 = np.roll(grid, -1, axis = 1)
    M6 = np.roll(np.roll(grid, 1, axis = 1), -1, axis = 0)
    M7 = np.roll(grid, -1, axis = 0)
    M8 = np.roll(np.roll(grid, -1, axis = 1), -1, axis = 0)
    return M0 + 2*M1 + 4*M2 + 8*M3 + 16*grid + 32*M5 + 64*M6 + 128*M7 + 256*M8'''

kernel = np.array([[1, 2, 4],
                    [8, 16, 32],
                    [64, 128, 256]])

def encoding(grid):
    return nd.convolve(grid, kernel, mode = 'wrap').astype(int)


def fitting(grid):
    right = np.roll(grid, -1, axis = 1) #(x, y) -> (x+1, y)
    down = np.roll(grid, -1, axis = 0) #(x, y) -> (x, y + 1)
    diag_down = np.roll(grid, (-1, -1), axis = (1, 0)) #(x, y) -> (x+1, y+1)
    diag_up = np.roll(grid, (-1, 1), axis = (1, 0)) #(x, y) -> (x+1, y-1)
    points = np.where(grid == right, -8, 0) + np.where(grid == down, -8, 0) + np.where(grid == diag_down, 8, -5) + np.where(grid == diag_up, 8, -5)
    
    return points.sum()


def select_parent(chromosomes, fitness_values, n_chrom = 5):
    idx = np.random.randint(0, len(chromosomes), n_chrom)
    best_idx = idx[np.argmax([fitness_values[i] for i in idx])]
    return chromosomes[best_idx]


def reproduction(parent1, parent2):
    if np.random.uniform(0, 1) < rep_chance:
        rep_point = np.random.randint(1, len(parent1))
        return np.concatenate([parent1[:rep_point], parent2[rep_point:]]), np.concatenate([parent2[:rep_point], parent1[rep_point:]])
    else:
        return parent1, parent2


def mutation(chromosome):
    mask = (np.random.uniform(0, 1, chromosome_size) < mutation_chance).astype(int)
    return chromosome ^ mask


#chromosomes and fitness values

chromosomes = [get_chromosome(chromosome_size) for _ in range(population_size)]
best_fitness_history = []
teoretical_max = 40000

for gen in range(generations):
    
    fitness_values = []

    for chromosome in chromosomes:
        fitness = 0
        for _ in range(n_test):
            
            grid = np.random.randint(0, 2, (world_size, world_size))

            for _ in range(steps):
                encoded = encoding(grid)
                grid = chromosome[encoded]
            
            fitness += fitting(grid)
        fitness_values.append(fitness / n_test)

    current_best = max(fitness_values)
    best_fitness_history.append(current_best)

    
    print(f'gen {gen} finished, best fitness: {current_best:.1f}')

    if current_best >= 39900: 
        print(f'optimal solution was achieved')
        break

    #creating new population
    new_population = []

    #best chromosome stays without mutation
    best_idx = np.argmax(fitness_values)
    new_population.append(chromosomes[best_idx].copy())

    #reproduction and mutation
    while len(new_population) < population_size:
        p1 = select_parent(chromosomes, fitness_values)
        p2 = select_parent(chromosomes, fitness_values)

        c1, c2 = reproduction(p1, p2)
        new_population.append(mutation(c1))

        if len(new_population) < population_size:
            new_population.append(mutation(c2))

    chromosomes = new_population



plt.figure(figsize = (12, 5))
plt.subplot(1, 3, 1)
plt.plot(best_fitness_history)
plt.title('Fitting function vs time')
plt.xlabel('Generation')
plt.ylabel('Max fitting function')
plt.grid(True, alpha = 0.5)


best_ever = chromosomes[np.argmax(fitness_values)]
test_grid = np.random.randint(0, 2, (world_size, world_size))
initial_grid = test_grid.copy()
for _ in range(steps):
    test_grid = best_ever[encoding(test_grid)]

plt.subplot(1, 3, 2)
plt.imshow(test_grid, cmap = 'binary')
plt.title('Best chromosome results')

plt.subplot(1, 3, 3)
plt.imshow(initial_grid, cmap = 'binary')
plt.title('Initial grid')

plt.tight_layout()
plt.show()





    






