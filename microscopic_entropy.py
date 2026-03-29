import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

L = 12
N = 50


def make_rule(rule_number):
    b = np.array(list(np.binary_repr(rule_number, 8)), dtype=int)
    return b[::-1]


def S_t(counts):
    total = sum(counts.values())
    S = 0
    for c in counts.values():
        p = c / total
        S -= p * np.log2(p)
    return S

def S_t(counts):
    total = sum(counts.values())
    p = np.array(list(counts.values())) / total
    return -np.sum(p * np.log2(p))


#irreversible
def evolve_irreversible(states, rule):
    left  = np.roll(states, 1)
    center = states
    right = np.roll(states, -1)
    idx = 4*left + 2*center + right
    return rule[idx]

def entropy_irreversible(rule_n):
    rule = make_rule(rule_n)
    all_states = np.array([list(map(int, np.binary_repr(i,L))) for i in range(2**L)], dtype=int)
    S_list = []

    for t in range(N):
        counts = Counter(tuple(row) for row in all_states)
        S_list.append(S_t(counts))
        all_states = np.array([evolve_irreversible(row, rule) for row in all_states])

    return S_list

#reversible
def entropy_reversible(rule_n):
    rule = make_rule(rule_n)

    #initial states
    all_curr = np.array([list(map(int, np.binary_repr(i,L))) for i in range(2**L)], dtype=int)
    all_prev = np.zeros_like(all_curr)
    

    phase_dict = Counter()
    for prev, curr in zip(all_prev, all_curr):
        key = tuple(np.concatenate([prev,curr]))
        phase_dict[key] = 1

    S_list = []

    for t in range(N):
        S_list.append(S_t(phase_dict))

        #evolution
        new_phase = Counter()
        for key, count in phase_dict.items():
            prev = np.array(key[:L])
            curr = np.array(key[L:])
            
            left = np.roll(curr,1)
            center = curr
            right = np.roll(curr,-1)
            idx = 4*left + 2*center + right
            evolved = rule[idx]

            next_curr = (evolved + prev) % 2
            new_key = tuple(np.concatenate([curr, next_curr]))
            new_phase[new_key] += count

        phase_dict = new_phase

    return S_list


S110 = entropy_irreversible(110)
S122R = entropy_reversible(122)

plt.figure(figsize=(12,5))
plt.suptitle('microscopic entropy S(t) for automata 110 and 122R on 12 cells')

plt.subplot(1,2,1)
plt.plot(S110)
plt.title("S(t) for automaton 110")
plt.xlabel("t")
plt.ylabel("S(t)")

plt.subplot(1,2,2)
plt.plot(S122R)
plt.title("S(t) for automaton 122R")
plt.xlabel("t")
plt.ylabel("S(t)")

plt.tight_layout()
plt.show()

#differences comment:
#automaton 110 shows a decrease in entropy over time, indicating a loss of information and evolution toward a more ordered state
#automaton 122R maintains a constant entropy level which is a predicted outcome because it is reversible so no information is lost