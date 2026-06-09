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



# Import necessary libraries
import math
from sympy import symbols, Eq, solve

# Function to format answers (3-digit format required by competition)
def format_answer(n):
    return str(n).zfill(3)  # Ensures output is always 3 digits

# 1ï¸�âƒ£ Solve Quadratic Equations
def solve_quadratic(a, b, c):
    x = symbols('x')
    eq = Eq(a*x**2 + b*x + c, 0)
    solutions = solve(eq, x)
    
    real_solutions = [sol.evalf() for sol in solutions if sol.is_real]
    if not real_solutions:
        return None
    
    valid_solutions = [int(sol) for sol in real_solutions if sol >= 0]
    return format_answer(valid_solutions[0] % 1000) if valid_solutions else None

# 2ï¸�âƒ£ Solve GCD Problems
def solve_gcd(a, b):
    return format_answer(math.gcd(a, b) % 1000)

# 3ï¸�âƒ£ Solve Modular Arithmetic Problems
def solve_modular(base, exponent, mod):
    return format_answer(pow(base, exponent, mod))

# Example Test Cases
print("Quadratic Solver:", solve_quadratic(1, -5, 6))  # x^2 - 5x + 6 = 0
print("GCD Solver:", solve_gcd(252, 105))
print("Modular Solver:", solve_modular(3, 7, 1000))


