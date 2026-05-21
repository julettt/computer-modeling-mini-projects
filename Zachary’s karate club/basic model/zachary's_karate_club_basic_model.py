import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation


D = 5
beta = 10
dt = 0.01
max_steps = 6000
steps_saved = 10


def init_graph():
    g_loc = nx.karate_club_graph()
    
    #nodes states
    for i in g_loc.nodes:
        g_loc.nodes[i]['state'] = 0.5
    g_loc.nodes[0]['state'] = 0     #node0 -> Mr. Hi
    g_loc.nodes[33]['state'] = 1    #node33 -> John A

    #edges weights
    for i, j in g_loc.edges:
        g_loc[i][j]['weight'] = 0.5

    return g_loc

def f(x):
    return (x - 0.25)**3

g = init_graph()
pos = nx.spring_layout(g, seed = 12)


steps = 0
frames = []

while steps < max_steps:
    
    steps += 1
    g_copy = g.copy()

    #node evolution
    for i in g.nodes:
        
        if i != 0 and i != 33:
            
            c_i = g.nodes[i]['state']

            c_sum = 0
            for j in g.neighbors(i):
                c_j = g.nodes[j]['state']
                w_ij = g[i][j]['weight']
                
                c_sum += (c_j - c_i) * w_ij
            
            d_c = D * c_sum * dt

            g_copy.nodes[i]['state'] += d_c

    #weights evolution
    for i, j, data in g.edges(data = True):

        w_ij = data.get('weight')
        c_i, c_j = g.nodes[i]['state'], g.nodes[j]['state']

        d_w = -beta * w_ij * (1 - w_ij) * f(np.abs(c_i - c_j)) * dt
        
        g_copy[i][j]['weight'] += d_w

    g = g_copy
    
    if steps % steps_saved == 0:
        frames.append(g.copy())


#animation
fig, ax = plt.subplots(figsize=(8, 8))

def animate(idx):

    ax.clear()

    frame = frames[idx]

    curr_step = steps_saved * idx
    
    ax.set_title(f'current step: {curr_step}/{max_steps}')

    nx.draw(frame, pos, with_labels = True,
        cmap = plt.cm.cool, vmin = 0, vmax = 1,
        node_color = [frame.nodes[i]['state'] for i in frame.nodes],
        edge_cmap = plt.cm.binary, edge_vmin = 0, edge_vmax = 1,
        edge_color = [frame[i][j]['weight'] for i, j in frame.edges], ax = ax)

plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

ani = animation.FuncAnimation(fig, animate, frames = len(frames), interval = 1, repeat = False)
ani.save(f'karate_club_{max_steps}_steps.mp4', writer = 'ffmpeg', fps = 10, dpi = 200, extra_args=['-preset', 'ultrafast'])



plt.show()
