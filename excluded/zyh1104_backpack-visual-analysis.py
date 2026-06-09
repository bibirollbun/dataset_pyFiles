pip install plotly


# Here are some of the packages that will be used

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import missingno as msno
#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


#load data
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


# Set id as index
train.set_index('id', inplace=True)
test.set_index('id', inplace=True)


#View the first 5 rows of data
train.head()


test.head()


#Returns the number of rows and columns for the data set
train.shape


test.shape


train.info()


test.info()


#View the number of missing values per column
train.isnull().sum()


#View the number of missing values per column
test.isnull().sum()


msno.bar(train)
plt.show()

msno.matrix(train)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(train.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()



# Get numerical and categorical columns
num_cols = list(train.select_dtypes(include=['number']).columns)
cat_cols = list(train.select_dtypes(exclude=['number']).columns)

# Print results
print("Numerical columns:")
print(num_cols)
print("\nCategorical columns excluding datetime columns:")
print(cat_cols)


#View statistical descriptions of numerical features, including the mean, standard deviation, minimum, maximum, and so on
train[num_cols].describe()


plt.figure(figsize=(16,12))
for i, col in enumerate(num_cols):
    plt.subplot(4, 4, i+1)
    sns.histplot(train[col], kde=True, color='blue')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


# Calculate the correlation coefficient matrix
correlation_matrix = train[num_cols].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', linewidths=0.5) 

plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# Make sure that 'Price' is listed in num_cols
if 'Price' not in num_cols:
    num_cols.append('Price')

# Calculate the correlation coefficient matrix
correlation_matrix = train[num_cols].corr()

# Gets the correlation with the Price
correlation_with_premium = correlation_matrix.loc[:, 'Price'].drop('Price', errors='ignore')

# Order the correlation coefficients
sorted_correlation = correlation_with_premium.sort_values(ascending=False)

# Draw the sorted bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x=sorted_correlation.values, y=sorted_correlation.index, palette="viridis")

plt.title('Correlation of Numerical Features with Price ', fontsize=16, fontweight='bold')
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(18,12))
for i, col in enumerate(cat_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(train[col], kde=True, color='blue')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


sns.set_style("whitegrid")

num_plots = len(cat_cols)
num_rows = (num_plots + 2) // 3

fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))

if num_rows == 1:
    axes = axes.reshape(1, -1)

for i, col in enumerate(cat_cols):
    counts = train[col].value_counts()
    total = counts.sum()
    
    row = i // 3
    col_index = i % 3
    
    ax = axes[row, col_index]
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom')

    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {col}', fontsize=16, fontweight='bold')

for j in range(num_plots, num_rows * 3):
    row = j // 3
    col_index = j % 3
    axes[row, col_index].axis('off')

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")  

sns.boxplot(
    x=train["Price"], 
    color="skyblue", 
    width=0.4,
    linewidth=1.5,
    flierprops=dict(marker="o", markersize=5, markerfacecolor="r")  
)


plt.title("Distribution of Price (Boxplot)", fontsize=14, pad=20)
plt.xlabel("Price", fontsize=12)
plt.xticks(fontsize=10)
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Compartments vs Price
sns.regplot(
    data=train, x="Compartments", y="Price", 
    ax=axes[0], scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}
)
axes[0].set_title("Compartments vs Price")

# Weight Capacity (kg) vs Price
sns.regplot(
    data=train, x="Weight Capacity (kg)", y="Price", 
    ax=axes[1], scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}
)
axes[1].set_title("Weight Capacity vs Price")

plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 20))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(4, 2, i)
    sns.violinplot(
        data=train, x=col, y="Price", 
        palette="pastel", cut=0 
    )
    plt.xticks(rotation=45)
    plt.title(f"{col} vs Price")

plt.tight_layout()
plt.show()


binary_cols = ['Laptop Compartment', 'Waterproof']

plt.figure(figsize=(10, 6))
for i, col in enumerate(binary_cols, 1):
    plt.subplot(1, 2, i)
    sns.barplot(
        data=train, x=col, y="Price", 
        estimator="mean", errorbar=None, 
        palette="Set2"
    )
    plt.title(f"Average Price by {col}")

plt.tight_layout()
plt.show()


color_cols = ['Pink', 'Gray', 'Blue', 'Red', 'Green', 'Black']

plt.figure(figsize=(12, 6))
sns.barplot(
    data=train[train["Color"].isin(color_cols)],  
    x="Color", 
    y="Price", 
    order=color_cols,  
    palette=[col.lower() for col in color_cols], 
    estimator="mean", 
    errorbar=None
)
plt.title("Average Price by Color")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(
    data=train, y="Brand", 
    order=train["Brand"].value_counts().index[:5],  # Take Top5 brands
    palette="viridis"
)
plt.title("Top 5 Brands by Count")
plt.show()


# Select Top5 brands and Top3 materials
top_brands = train["Brand"].value_counts().index[:5]
top_materials = train["Material"].value_counts().index[:3]
filtered_data = train[train["Brand"].isin(top_brands) & train["Material"].isin(top_materials)]

plt.figure(figsize=(14, 8))
sns.boxplot(
    data=filtered_data, 
    x="Brand", y="Price", hue="Material",
    palette="Blues"
)
plt.title("Price Distribution by Brand and Material")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 8))

scatter = sns.scatterplot(
    data=train,
    x="Weight Capacity (kg)",
    y="Compartments",
    hue="Price",
    palette="viridis",
    size=20,  
    alpha=0.7
)

plt.colorbar(scatter.collections[0], label="Price")
plt.title("Weight Capacity vs Compartments (Color = Price)")
plt.show()


for cat_col in cat_cols:
    sns.pairplot(train[num_cols + [cat_col]], hue=cat_col, diag_kind='kde', corner=True)
    plt.title(f"Pairplot with hue={cat_col}")
    plt.show()

