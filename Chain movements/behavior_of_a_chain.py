import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

k = 10**3
l_0 = 1.0
k_b = 10**2
N = 100
kbT = 0.1

steps = 1000
s = 0.05

mode = 'animation' #'animation' or 'snapshots'

def E_total(pos_x, pos_y):
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    lengths = np.sqrt(dx**2 + dy**2)
    Es = 0.5 * k * np.sum((lengths - l_0)**2)
    
    ux, uy = pos_x[:-2] - pos_x[1:-1], pos_y[:-2] - pos_y[1:-1]
    wx, wy = pos_x[2:] - pos_x[1:-1], pos_y[2:] - pos_y[1:-1]
    
    u_norm = np.sqrt(ux**2 + uy**2)
    w_norm = np.sqrt(wx**2 + wy**2)

    cos_theta = (ux*wx + uy*wy) / (u_norm * w_norm + 1e-12)
    Eb = k_b * np.sum(1 + cos_theta)
    
    return Es + Eb

  
def monte_carlo(epsilon, one_by_one = True):
    L_0 = (N-1) * l_0
    L = L_0 * (1+epsilon)
    
    #points initial positions
    x = np.linspace(0, L, N)
    y = np.random.normal(0, 0.01, N)

    #fixed bounds
    y[0] = y[-1] = 0 

    current_E = E_total(x, y)
    history = []
    
    for step in range(steps):
        if one_by_one:
            for i in range(1, N - 1):
                old_x, old_y = x[i], y[i]
                
                x[i] += np.random.uniform(-s, s)
                y[i] += np.random.uniform(-s, s)
                
                new_E = E_total(x, y)
                delta_E = new_E - current_E
                
                #accept?
                if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / kbT):
                    current_E = new_E
                else:
                    x[i], y[i] = old_x, old_y
    
        else:
            old_x = x.copy()
            old_y = y.copy()
            
            noise_x = np.random.uniform(-s, s, N)
            noise_y = np.random.uniform(-s, s, N)
            
            noise_x[0] = noise_x[-1] = 0
            noise_y[0] = noise_y[-1] = 0
            
            x += noise_x
            y += noise_y
            
            new_E = E_total(x, y)
            delta_E = new_E - current_E
            
            #accept?
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / kbT):
                current_E = new_E
            else:
                x = old_x.copy()
                y = old_y.copy()

        if step % 10 == 0:
                history.append((x.copy(), y.copy()))
            
    return history


epsilon = -0.1

#simulation
history1 = monte_carlo(epsilon, one_by_one = True)
history2 = monte_carlo(epsilon, one_by_one = False)


#presenting results - mode animation or snapshots
if mode == 'animation':

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))

    line1, = ax1.plot([], [], 'r-', lw=1.5)
    line2, = ax2.plot([], [], 'b-', lw=1.5)

    for ax, title in zip([ax1, ax2], ['changing position one by one in each step', 'changing position all at once in each step']):
        ax.set_xlim(-5, (N-1) * l_0 * (1+epsilon) + 5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def update(frame):
        x1, y1 = history1[frame]
        x2, y2 = history2[frame]
        line1.set_data(x1, y1)
        line2.set_data(x2, y2)
        return line1, line2

    ani = FuncAnimation(fig, update, frames=len(history1), interval=50, blit=True)

    plt.tight_layout()
    plt.show()

elif mode == 'snapshots':

    frames = [0, len(history1)//3, (2*len(history1))//3, len(history1) - 1]
    
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=False, sharey=False)
    
    column_titles = ['Changing position one by one in each step', 'Changing position all at once in each step']

    for row_i, idx in enumerate(frames):
        
        real_step = idx * 10

        data = [history1[idx], history2[idx]]
        colors = ['red', 'blue']
        
        for col_i in range(2):
            ax = axes[row_i, col_i]
            curr_x, curr_y = data[col_i]
            
            ax.plot(curr_x, curr_y, color=colors[col_i], lw=1.2)
            ax.grid(True, alpha=0.3)
            
            if col_i == 0:
                ax.set_ylim(-2, 2)
            else:
                ax.set_ylim(-0.1, 0.1)
            ax.set_xlim(-5, (N-1) * l_0 * (1+epsilon) + 5)
            
            if row_i == 0:
                ax.set_title(column_titles[col_i])
            if col_i == 0:
                ax.set_ylabel(f'Step {real_step}')

    plt.tight_layout()
    plt.show()