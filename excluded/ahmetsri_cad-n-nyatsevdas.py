import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



def period():
    A100, A120, A150 = 100, 120, 150
    cycle= 600 #ekok(100, 120, 150)
    flightDs= set()
    for day in range(0, cycle, A100):
        flightDs.add(day)

