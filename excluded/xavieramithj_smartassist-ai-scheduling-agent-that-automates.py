# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pip install pulp


from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatus, value
import pandas as pd

# --- 1. Problem Setup ---

SLOTS = ['Mon 9am', 'Mon 11am', 'Tue 9am', 'Tue 11am']
CRITICAL_THRESHOLD = 4

PRIORITY = {'Anna (CEO)': 5, 'Ben (PM)': 4, 'Cathy (Sales)': 2}

AVAILABILITY = pd.DataFrame({
    'Anna (CEO)': [0, 1, 1, 1],
    'Ben (PM)':   [1, 1, 0, 1],
    'Cathy (Sales)':[1, 1, 1, 0]
}, index=SLOTS)

PREFERENCE_SCORE = pd.Series([5, 10, 5, 10], index=SLOTS) 

RESOURCE_FREE = pd.Series([1, 1, 0, 1], index=SLOTS)

# --- 2. Optimization Model ---

problem = LpProblem("SmartAssist_Scheduler", LpMaximize)

X = LpVariable.dicts("Slot", SLOTS, 0, 1, LpBinary)

objective_components = []

for s in SLOTS:
    objective_components.append(X[s] * PREFERENCE_SCORE[s])
    
    for p, prio in PRIORITY.items():
        if AVAILABILITY.loc[s, p] == 1:
            objective_components.append(X[s] * prio)

problem += lpSum(objective_components), "Total_Meeting_Quality_Score"

problem += lpSum(X.values()) == 1, "Select_One_Slot"

for s in SLOTS:
    for p, prio in PRIORITY.items():
        if prio >= CRITICAL_THRESHOLD and AVAILABILITY.loc[s, p] == 0:
            problem += X[s] == 0, f"Critical_Conflict_Check_{p}_{s}"

for s in SLOTS:
    if RESOURCE_FREE[s] == 0:
        problem += X[s] == 0, f"Resource_Conflict_Check_{s}"

# --- 3. Solution Execution ---
problem.solve()

# --- 4. Accurate Output ---

print("--- SmartAssist Final Schedule ---")
print(f"Status: {LpStatus[problem.status]}") 

# PuLP code for Optimal status is 1. This avoids the AttributeError.
if problem.status == 1: 
    chosen_slot = next(s for s in SLOTS if value(X[s]) == 1)
    
    print(f"\nâœ… OPTIMAL TIME FOUND: **{chosen_slot}**")
    print(f"Total Quality Score: {value(problem.objective):.0f}")
    
    busy = [p for p in PRIORITY if AVAILABILITY.loc[chosen_slot, p] == 0 and PRIORITY[p] < CRITICAL_THRESHOLD]
    
    if busy:
        print(f"\nâš ï¸� Acceptable Compromise:")
        print(f"* Unavailability of non-critical attendees: **{', '.join(busy)}**")
        print("* All critical participants (CEO, PM) are available.")
    else:
        print("\nğŸ�‰ All participants and resources are available.")
    
elif problem.status == -1: 
    print("\nâ�Œ NO FEASIBLE SOLUTION. Adaptive Agent Required.")
    print("Action: No slot met all hard constraints (Critical attendee and resource availability).")
    
else:
    print("\nâš ï¸� Solver could not find a solution.")

