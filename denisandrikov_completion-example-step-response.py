!pip install control --quiet
import control
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


T = 0.5
ksi = 0.05
K = 2
W = control.tf([K],[T**2, 2*ksi*T, 1])
t, y = control.step_response(W, 50)
plt.plot(t,y)
plt.grid(which='minor', alpha=0.2)
plt.minorticks_on()
plt.grid(which='major', alpha=0.8)

