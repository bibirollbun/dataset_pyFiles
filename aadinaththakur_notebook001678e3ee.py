import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix



def f(x):
    z = [[0]*9 for _ in range(9)]
    for i,j in [(0,0),(0,6),(6,0),(6,6)]:
        for r in range(3):
            for c in range(3):
                z[i+r][j+c] = x[r][c]
    return z

# Test case
input_grid = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
]

expected_output = [
    [1, 0, 1, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1]
]

print(f(input_grid) == expected_output)


import zipfile
from io import BytesIO

# Define your function as a string
code = """def f(x):
    z = [[0]*9 for _ in range(9)]
    for i,j in [(0,0),(0,6),(6,0),(6,6)]:
        for r in range(3):
            for c in range(3):
                z[i+r][j+c] = x[r][c]
    return z
"""

# Create an in-memory ZIP file
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w") as zipf:
    zipf.writestr("solution.py", code)

# Save the ZIP to disk
with open("submission.zip", "wb") as f:
    f.write(zip_buffer.getvalue())

