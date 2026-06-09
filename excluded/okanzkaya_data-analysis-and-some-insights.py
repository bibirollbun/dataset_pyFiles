import pandas as pd

# Load dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")

# Basic overview
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)



df.describe()


import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
numeric_cols = df.select_dtypes(include='number')
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


