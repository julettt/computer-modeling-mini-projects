import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

#mode (survive/birth)

#Mazectric (1234/3)
#Flakes (012345678/3)
#Serviettes (/234)
#WalledCities (2345/45678)
#Life (23/3)

survive = [1, 4, 2, 3, 5, 6, 7, 8]
birth = [3]


def neighbours(grid):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    return ss.convolve2d(grid, kernel, mode = 'full', boundary = 'fill', fillvalue = 0)[1:-1, 1:-1]


def determine_state(G):
    N = neighbours(G)
    G = np.where(((G == 1) & np.isin(N, survive)) | ((G == 0) & np.isin(N, birth)), 1, 0)
    return G

#state = np.random.randint(0,2,(200, 200))


state = np.zeros((200, 200))
#state[100:102, 100:102] = 1 #Serviette
state[70:130, 70:130] = 1 #Flakes, Life, Mazectric

#evolution
frames = []
for _ in range(300):
    state = determine_state(state)
    frames.append(state)


fig, ax = plt.subplots()
ax.axis('off') 
ims = []
for f in frames:
    im = ax.imshow(f, cmap='binary', animated=True)
    ims.append([im])

animation = ArtistAnimation(fig, ims, interval=10, blit=True)
plt.show()

