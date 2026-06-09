import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as ws
from scipy.stats import chi2_contingency


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df.head()


len(df)


df.isnull().any()


def plot_categorical_defaulters(df, column, target="loan_paid_back"):
    # Compute counts
    ctab = pd.crosstab(df[column], df[target])
    
    # Add % Defaulters (assuming 0 = default, 1 = paid back)
    ctab["% Defaulters"] = (ctab[0] / (ctab[0] + ctab[1]) * 100).round(2)
    
    # Display the table
    print(f"\n=== {column.upper()} ===")
    print(ctab)
    
    # Plot countplot
    plt.figure(figsize=(20, 4))
    sns.countplot(data=df, x=column, hue=target)
    plt.title(f"Loan Repayment by {column}")
    plt.xticks(rotation=45)
    plt.show()


plot_categorical_defaulters(df, "gender")


plot_categorical_defaulters(df, "marital_status")


plot_categorical_defaulters(df, "education_level")


plot_categorical_defaulters(df, "loan_purpose")


plot_categorical_defaulters(df, "grade_subgrade")

