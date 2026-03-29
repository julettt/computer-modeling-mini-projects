import numpy as np
from PIL import Image
import scipy.optimize as so
import math
import matplotlib.pyplot as plt 


L = 500
N = 15000
rule_n = 122
n = 41
reversible = True

m = 5
T = 300

#rule dictionary binary version
rule_binary = np.array([int(x) for x in format(rule_n, '08b')[::-1]], dtype=np.uint8)


#initial state
initial_str = '0' * ((L-n) // 2) + '1' * n + '0' * (L - (L-n) // 2 - n)
initial_state = np.array([int(i) for i in initial_str])

states = np.zeros((N, L), dtype=np.uint8)
states[0] = initial_state
states[1] = initial_state


#evolution
for t in range(1, N-1):
    left = np.roll(states[t], 1)
    center = states[t]
    right = np.roll(states[t], -1)

    ix = (left * 4 + center * 2 + right)
    
    evolved = np.array(rule_binary)[ix]
    
    if reversible:
        states[t+1] = (evolved + states[t-1]) % 2
    else:
        states[t+1] = evolved

#entropy
groups = states.reshape(N, L//m, m)
k_values = np.sum(groups, axis=2)

#for each time step
N_t = []

for t in range(N):
    group_sum = 0
    for k in k_values[t]:
        N_mk = math.comb(m, k)
        group_sum += np.log(N_mk)
    N_t.append(group_sum)

#entropy
S_t = []
for t in range(N-T):
    s = np.sum(N_t[t : t + T])
    S_t.append(s/T)

plt.figure(figsize=(10, 4))
plt.plot(S_t)
plt.title("Entropy")
plt.xlabel("t")
plt.ylabel("S(t)")
plt.show()


#img_data = (1 - states) * 255
#img_data = img_data.T
#img = Image.fromarray(img_data)
#img.show()