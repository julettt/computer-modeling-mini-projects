import numpy as np
import matplotlib.pyplot as plt

N = 5 #grid size


patterns = {
    'A': np.array([[0, 1, 1, 1, 0],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 0, 0, 0, 1]]),

    'Z': np.array([[1, 1, 1, 1, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 1, 1, 1, 1]])
}


train_data = np.array([np.where(p.flatten() == 0, -1, 1) for p in patterns.values()])
n_patterns, n_neurons = train_data.shape

#weight matrix
W = (train_data.T @ train_data) / n_patterns
np.fill_diagonal(W, 0)

def calculate_energy(s, weights):
    return -0.5 * np.dot(s.T, weights @ s)


def encode_pattern(test_spin, weights, max_iter=500, n_restarts=10):
    best_s, best_energy_history = None, None
    best_final_energy = np.inf

    for _ in range(n_restarts):
        s = np.copy(test_spin)
        energy_history = [calculate_energy(s, weights)]
        n = len(s)

        for _ in range(max_iter):
            order = np.random.permutation(n)
            changed = False

            for i in order:
                new_val = np.sign(weights[i] @ s)
                if new_val == 0:
                    new_val = s[i]
                if new_val != s[i]:
                    s[i] = new_val
                    changed = True

            energy_history.append(calculate_energy(s, weights))

            if not changed:
                break

        if energy_history[-1] < best_final_energy:
            best_final_energy = energy_history[-1]
            best_s = s
            best_energy_history = energy_history

    return best_s, best_energy_history


def find_closest_pattern(encoded, train_data, pattern_names):
    overlaps = {name: float(np.dot(encoded, pat)) / n_neurons
                for name, pat in zip(pattern_names, train_data)}
    for name, pat in zip(pattern_names, train_data):
        if np.array_equal(encoded, pat):
            return name, overlaps
    best = max(overlaps, key = lambda k: overlaps[k])
    return best, overlaps


test_letter2 = np.array([
              [0, 1, 1, 1, 1],
              [1, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1],
              [1, 1, 1, 0, 1]])


test_letter = np.array([
              [0, 1, 1, 0, 0],
              [1, 0, 0, 0, 1],
              [0, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 0, 0, 0, 0]])

test_letters = {
    'Z': np.array([
              [0, 1, 1, 1, 1],
              [1, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1],
              [1, 1, 1, 0, 1]]),

    'A': np.array([
              [0, 1, 1, 0, 0],
              [1, 0, 0, 0, 1],
              [0, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 0, 0, 0, 0]])
}

pattern_names = list(patterns.keys())

fig, axes = plt.subplots(len(test_letters), 3, figsize=(10, 4 * len(test_letters)))

for row, (test_name, test_letter) in enumerate(test_letters.items()):
    test_s = np.where(test_letter.flatten() == 0, -1, 1)
    encoded_image, energy = encode_pattern(test_s, W)
    letter_image = np.where(encoded_image.reshape(N, N) == -1, 0, 1)

    recognized, overlaps = find_closest_pattern(encoded_image, train_data, pattern_names)
    print(f"tested: '{test_name}': detected: '{recognized}'")

    axes[row, 0].imshow(test_letter)
    axes[row, 0].set_title('test image', fontsize = 8)
    axes[row, 0].axis('off')

    axes[row, 1].imshow(letter_image)
    axes[row, 1].set_title(f'detected letter: {recognized}', fontsize = 8)
    axes[row, 1].axis('off')

    axes[row, 2].plot(energy, 'C0o')
    axes[row, 2].plot(energy, '--', alpha=0.5)
    axes[row, 2].set_title('energy', fontsize = 8)

plt.tight_layout()
plt.show()