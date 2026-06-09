# Import thÆ° viá»‡n cáº§n thiáº¿t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # áº¨n táº¥t cáº£ cáº£nh bÃ¡o



# Load dataset
df_train = pd.read_csv("/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv")
df_test = pd.read_csv("/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv")
# df_test.head()


print("Train data - num column: ", len(df_train.columns))
print("Train data - num row: ", len(df_train))
print("ğŸ“Œ Output column name:", df_train.columns[-1])
df_train.info()
#Test data
print("Test data - num column: ", len(df_test.columns))
print("Test data - num row: ", len(df_test))
df_test.info()



# Thá»‘ng kÃª train data
def summarize_columns(df):
    total = len(df)

    result = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Count": df.isnull().sum(),
        "Missing %": df.isnull().mean() * 100,
    })

    result["Unique Values"] = [
        df[col].nunique()
        for col in df.columns
    ]

    print(result)

summarize_columns(df_train)
df_train.describe()


kde_columns = ['latitude', 'longitude', 'floorAreaSqM']
hist_columns = ['sale_month', 'sale_year', 'price']
bar_columns = ['bedrooms', 'bathrooms', 'livingRooms', 'currentEnergyRating', 'tenure']
n_plots = len(kde_columns) + len(hist_columns) + len(bar_columns)
n_cols = 3
n_rows = (n_plots // n_cols) + (n_plots % n_cols > 0)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))


axes = axes.flatten()

for i, column in enumerate(kde_columns):
    sns.kdeplot(data=df_train, x=column, fill=True, ax=axes[i])  # Ä�iá»�u chá»‰nh Ä‘á»™ mÆ°á»£t
    axes[i].set_title(f'KDE of {column}')
    axes[i].set_xlabel(column)

for i, column in enumerate(bar_columns, start=len(kde_columns)):
    sns.countplot(data=df_train, x=column, ax=axes[i])
    axes[i].set_title(f'Bar plot of {column}')

for i, column in enumerate(hist_columns, start=len(kde_columns) + len(bar_columns)):
    sns.histplot(data=df_train, x=column, kde=True, ax=axes[i])
    axes[i].set_title(f'Histogram of {column}')

for i in range(n_plots, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# Táº¡o cá»™t price Ä‘Ã£ log-transform (log10)
log_price = np.log10(df_train['price'])

plt.figure(figsize=(10, 6))
sns.histplot(data=log_price, kde=True)
plt.title('Histogram of log10(Price)')
plt.xlabel('log10(Price)')
plt.tight_layout()
plt.show()


numeric_cols = df_train.select_dtypes(include=['number']).columns

for col in numeric_cols:
    plt.figure(figsize=(10, 2))
    sns.boxplot(x=df_train[col])
    plt.title(f'Boxplot of {col}')
    plt.show()



plt.figure(figsize=(10, 2))
sns.boxplot(x=np.log10(df_train['price']))
plt.title("Box plot of log(price)")
plt.show()


numeric_features = [col for col in df_train.select_dtypes(include=['number']).columns if col != 'ID']
correlation_matrix = df_train[numeric_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Ma tran hiep tuong quan')
plt.show()



num_features = df_train.select_dtypes(include='number').drop(columns=['price', 'ID'], errors='ignore').columns
n = len(num_features)
total_plots = (n * (n - 1)) // 2
cols = 3
rows = (total_plots + cols - 1) // cols

fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
ax = ax.flatten()

plot_idx = 0
for i in range(n):
    for j in range(i + 1, n):
        sns.scatterplot(data=df_train, x=num_features[i], y=num_features[j], ax=ax[plot_idx], alpha=0.3)
        ax[plot_idx].set_title(f'{num_features[i]} vs {num_features[j]}')
        plot_idx += 1


plt.tight_layout()
plt.suptitle("Feature-Feature", fontsize=16, y=1.02)
plt.show()


import math
num_features = df_train.select_dtypes(include='number').drop(columns=['price', 'ID', 'log_price'], errors='ignore').columns
cols = 3
rows = math.ceil(len(num_features) / cols)

fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
ax = ax.flatten()

for i, col in enumerate(num_features):
    sns.scatterplot(data=df_train, x=col, y='price', ax=ax[i], alpha=0.3)
    ax[i].set_title(f'{col} vs Price')

plt.tight_layout()
plt.suptitle("Feature - Output", fontsize=18, y=1.02)
plt.show()


df_train['sale_time'] = df_train['sale_year'] + (df_train['sale_month'] - 1) / 12
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_train, x='sale_time', y='price', estimator='mean', ci=None, color='blue')
plt.title('Price - Time (M/Y)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.tight_layout()
plt.show()



cat_cols = ['tenure', 'currentEnergyRating']
num_cols = ['bedrooms', 'bathrooms', 'livingRooms', 'sale_month']

n_plots = len(cat_cols) * len(num_cols)
n_cols = 2
n_rows = (n_plots + 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
axes = axes.flatten()

plot_idx = 0
for cat in cat_cols:
    for num in num_cols:
        ax = axes[plot_idx]
        count_data = df_train.groupby([cat, num]).size().reset_index(name='count')
        count_data['log_count'] = np.log10(count_data['count'])
        sns.barplot(data=count_data, x=num, y='log_count', hue=cat, ax=ax)
        ax.set_title(f"log10(count) theo {num} vÃ  {cat}")
        ax.set_ylabel("log10(count)")
        plot_idx += 1

plt.tight_layout()
plt.show()



df_train['log_price'] = np.log10(df_train['price'])

cat_cols = ['tenure', 'currentEnergyRating', 'propertyType']
num_cols = ['log_price', 'floorAreaSqM', 'sale_year']
fig, axes = plt.subplots(len(cat_cols), len(num_cols), figsize=(5 * (len(num_cols) + 2), 5 * len(cat_cols)))

for i, cat_col in enumerate(cat_cols):
    for j, num_col in enumerate(num_cols):
        ax = axes[i, j]
        sns.violinplot(data=df_train, x=cat_col, y=num_col, scale='width', ax=ax)
        ax.set_title(f"{num_col} theo {cat_col}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

plt.tight_layout()
plt.show()

