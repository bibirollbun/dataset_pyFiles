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


import pandas as pd
import matplotlib.pyplot as plt


def load_dataset():
    train_df=pd.read_csv("/kaggle/input/quitting-smoking-bgu-2025/train.csv")
    test_df=pd.read_csv("/kaggle/input/quitting-smoking-bgu-2025/test.csv")
    return train_df , test_df
  


train_df , test_df = load_dataset()


train_df.info()


test_df.info()


print("shape of train data ", train_df.shape)
print("shape of test data ", test_df.shape)


import matplotlib.pyplot as plt
import seaborn as sns


# Creating bins for age groups
train_df['age_group'] = pd.cut(train_df['age'], bins=[10, 20, 30, 40, 50, 60, 70], labels=['10-20', '20-30', '30-40', '40-50', '50-60', '60-70'])

# Grouping by age group and calculating mean
age_cholesterol_avg = train_df.groupby('age_group')['Cholesterol'].mean().reset_index()
age_height_avg = train_df.groupby('age_group')['height(cm)'].mean().reset_index()

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Age vs. Cholesterol
sns.lineplot(data=age_cholesterol_avg, x='age_group', y='Cholesterol', marker='o', ax=axes[0])
axes[0].set_title('Cholesterol by Age')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Cholesterol')

# Age vs. Height
sns.lineplot(data=age_height_avg, x='age_group', y='height(cm)', marker='o', ax=axes[1])
axes[1].set_title('Height by Age')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Height (cm)')

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Define the scatter plot pairs
scatter_pairs = [
    ("height(cm)", "weight(kg)"),
    ("waist(cm)", "Cholesterol"),
    ("age", "HDL"),
    ("systolic", "LDL"),
    ("fasting blood sugar", "hemoglobin"),
    ("age", "serum creatinine"),
]

# Calculate the number of rows and columns for the subplots
num_rows = 2  # You can adjust this as needed
num_cols = 3  # You can adjust this as needed

# Create the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Flatten the axes array to easily iterate over the subplots
axes = axes.flatten()

# Iterate through the scatter plot pairs and create the plots
for i, (x_col, y_col) in enumerate(scatter_pairs):
    ax = axes[i]  # Get the current subplot axis
    sns.scatterplot(data=train_df, x=x_col, y=y_col, hue="smoking", ax=ax, alpha=0.7)  # Hue added if categorical column exists
    
    # Add a regression line
    # sns.regplot(data=train_df, x=x_col, y=y_col, ax=ax, scatter=False, line_kws={"color": "red"})

    # Compute correlation coefficient
    corr = train_df[[x_col, y_col]].corr().iloc[0, 1]
    ax.annotate(f"r = {corr:.2f}", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=12, color="blue")

    # Set title, labels, and grid
    ax.set_title(f"{x_col} vs. {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True)

plt.tight_layout()
plt.show()


# Facet Plots
scatter_pairs = [
    ("height(cm)", "weight(kg)"),
    ("waist(cm)", "Cholesterol"),
    ("age", "HDL"),
    ("systolic", "LDL"),
    ("fasting blood sugar", "hemoglobin"),
    ("age", "serum creatinine"),
]

g = sns.FacetGrid(train_df, col='smoking', height=5)
g.map_dataframe(sns.scatterplot, x='height(cm)', y='weight(kg)').set_titles('Height vs. Weight (Smoking={col_name})')

g = sns.FacetGrid(train_df, col='dental caries', height=5)
g.map_dataframe(sns.scatterplot, x='age', y='Cholesterol').set_titles('Age vs. Cholesterol (Dental Caries={col_name})')

g = sns.FacetGrid(train_df, col='hearing(left)', height=5)
g.map_dataframe(sns.scatterplot, x='systolic', y='relaxation').set_titles('Systolic vs. Relaxation (Hearing Left={col_name})')

plt.show()


train_df.head()


sns.histplot(data=train_df, x="age", bins=30, kde=True)
plt.show()



sns.kdeplot(data=train_df, x="serum creatinine", fill=True)
plt.show()



sns.swarmplot(data=train_df, x="smoking", y="Cholesterol")
plt.show()




sns.stripplot(data=train_df, x="dental caries", y="weight(kg)", jitter=True)
plt.show()



sns.boxplot(data=train_df, x="smoking", y="HDL")
plt.show()



sns.violinplot(data=train_df, x="smoking", y="LDL", hue="dental caries", split=True)
plt.show()



sns.barplot(data=train_df, x="smoking", y="ALT")
plt.show()



sns.pointplot(data=train_df, x="Urine protein", y="Gtp", hue="smoking")
plt.show()



import numpy as np
corr_matrix = train_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()



sns.clustermap(corr_matrix, cmap="coolwarm", annot=True)
plt.show()


