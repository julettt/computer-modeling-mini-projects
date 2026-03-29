import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

def neighbours(grid):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    return ss.convolve2d(grid, kernel, mode = 'full', boundary = 'fill', fillvalue = 0)[1:-1, 1:-1]


def determine_state(G):
    N = neighbours(G)
    G = np.where(((G == 1) & ((N == 3) | (N == 2))) | ((G == 0) & (N == 3)), 1, 0)
    return G

state = np.random.randint(0,2,(256,512))

#evolution
frames = []
for _ in range(50):
    state = determine_state(state)
    frames.append(state)


fig, ax = plt.subplots()
ax.axis('off') 
ims = []
for f in frames:
    im = ax.imshow(f, cmap='binary', animated=True)
    ims.append([im])

animation = ArtistAnimation(fig, ims, interval=100, blit=True)
plt.show()

