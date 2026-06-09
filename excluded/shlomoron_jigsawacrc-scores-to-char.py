import numpy as np
import string
import matplotlib.pyplot as plt
import pickle


tick1 = 0.135-0.124
tick2 = 0.149-0.124
tick3 = 0.177-0.124
tick4 = 0.224-0.124
tick5 = 0.309-0.124
tick6 = 0.491-0.124


0.309-0.124


ticks = [0]
ticks_keys = [[0]]
for i,tick in enumerate([tick1,tick2,tick3,tick4,tick5,tick6]):
    curr_ticks = [x+tick for x in ticks]
    curr_ticks_keys = [[i+1]+x for x in ticks_keys]
    ticks = ticks+curr_ticks
    ticks_keys = ticks_keys+curr_ticks_keys
ticks = np.asarray(ticks)+0.124


# Concatenate the sets and convert to a list
printable_chars = list(string.ascii_lowercase + string.digits)+[' ', '\n', '.', ',', "'", '"', '!', '?', ':', '(', ')', '[', ']']
print(printable_chars)
print(len(printable_chars))


indices = [i for i in range(64)]
indices = [x for x in indices if (x+1)%16!=0 and x!=32]
ticks = ticks[indices]
ticks_keys = [ticks_keys[x] for x in indices]


plt.scatter(range(len(ticks)), ticks)


for i in range(len(printable_chars)):
    if printable_chars[i] == ' ':
        print(f'{round(ticks[i], 4)}: Space')
    elif printable_chars[i] == '\n':
        print(f'{round(ticks[i], 4)}: \\n')
    else:
        print(f'{round(ticks[i], 4)}: {printable_chars[i]}')


pickle.dump(ticks, open('ticks.p', 'bw'))
pickle.dump(ticks_keys, open('ticks_keys.p', 'bw'))
pickle.dump(printable_chars, open('printable_chars.p', 'bw'))

