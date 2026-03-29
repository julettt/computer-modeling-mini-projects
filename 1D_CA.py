import numpy as np
from PIL import Image, ImageDraw

#input
L = 600
rule_n = 30
reversible =False
N = 600

k = 1
initial_str = '0' * ((L-k) // 2) + '1' * k + '0' * (L - (L-k) // 2 - k)
#initial_str = ''.join(np.random.choice(['0', '1']) for _ in range(L))

current_state = np.array([int(i) for i in initial_str])


#rule dictionary
b = format(rule_n, '08b')
patterns = [(1,1,1), (1,1,0), (1,0,1), (1,0,0), (0,1,1), (0,1,0), (0,0,1), (0,0,0)]
rule_dict = {p: int(bit) for p, bit in zip(patterns, b)}

states = np.zeros((N, L))
states[0] = current_state
states[1] = current_state

#evolution
for t in range(N-1):
  current = states[t]
  
  for i in range(L):
    neighborhood = (current[(i - 1) % L], current[i], current[(i + 1) % L])
    evolved = rule_dict[neighborhood]
  
    if reversible:
      previous = states[t-1, i]
      states[t+1, i] = (evolved + previous) % 2
    else:
      states[t+1, i] = evolved

img = Image.new("RGB",(L, N),(255,255,255))
draw = ImageDraw.Draw(img)
for y in range(N):
  for x in range(L):
    if states[y][x]: draw.point((x,y),(0,0,0))
img.show()