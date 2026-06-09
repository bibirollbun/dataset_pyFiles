# Variables give a name to data
a = 3            # int
b = 2.5          # float
c = 8.87
name = "Alice"   # string str
nums = [1, 2, 3] # list 0, 1, 2

print(a + b)          # math
print(a + b + c)
print(name.upper())   # call a function on a string
print(nums[2])        # list indexing

# Always check the type when confused
print(type(a), type(name), type(nums))



# import [library_name] as [nickname]

import pandas as pd
import matplotlib.pyplot as plt 
import os


# STEP 1 — Point to your CSV file.
# Example (YOU must change this to your path if you have RSNA data):
# /kaggle/input/rsna-intracranial-aneurysm-detection/train.csv
CSV_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"

# STEP 2 — Try to read your CSV
df = pd.read_csv(CSV_PATH) # DataFrame
print("Loaded CSV from:", CSV_PATH)

df.head(5)  # show the first 5 rows


print("shape (rows, cols):", df.shape)
print("columns:", df.columns)
print("columns as list:", list(df.columns))



# # Look at a few rows
# display(df.head(3))       # first 3 rows
# print(df.head(3))


# display(df.sample(10))     # random 3 rows
# # print(df.sample(3))


# # Choose a column to count, you can change the target column
# target_col = "Aneurysm Present"
# # target_col = "PatientSex" 
# # target_col = "Modality"

# counts = df[target_col].value_counts()
# print(counts)

# # Plot a simple bar chart
# plt.figure(figsize=(5,4)) # Change these values, e.g., (10, 3), (4, 8), what do you see?
# counts.plot(kind="bar")
# plt.title(f"{target_col} counts") # Also, try to change the title of the figure
# plt.xlabel("target_col") 
# # plt.xlabel("Aneurysm Present")
#     # what's the advantage of the first way?
#     # Now, let's try to change the target_col.
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()


# Don't panic—this is on purpose to show the error.
# print(score)  # score is not defined yet


score = 95
print(score)


# "age: " + 12  # string + integer → not allowed



str(12)


"age: " + str(12)


# arr = np.array([1, 2, 3]) 


import numpy as np
arr = np.array([1, 2, 3])

# # Now it works! Let's print it to be sure.
print(arr)



# pd.read_csv("/no/such/folder/train.csv")




