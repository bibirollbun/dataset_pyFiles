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


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


df_train = pd.read_csv("/kaggle/input/introduction-to-data-secience-2025-competition/train.csv")
df_test = pd.read_csv("/kaggle/input/introduction-to-data-secience-2025-competition/test.csv")


df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()


df_train.shape,df_test.shape


df_train.info()


df_train.head(3)


df_test.info()


df_test.head(3)


df_train.isnull().sum(),df_test.isnull().sum()


df_train.describe()


import matplotlib.pyplot as plt
import seaborn as sns
import math

# Choose how many features to plot (or all)
selected_features = df_train.columns

# Automatically set rows and columns based on number of features
ncols = 2
nrows = math.ceil(len(selected_features) / ncols)

# Bigger figure size for better readability
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 3))
fig.suptitle('Boxplots of Selected Features', fontsize=20, weight='bold')
axes = axes.flatten()

# Plot boxplots for each feature
for i, col in enumerate(selected_features):
    sns.boxplot(y=df_train[col], ax=axes[i], color='skyblue')
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove unused subplots if any
for j in range(len(selected_features), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout for neat spacing
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()



set(df_train.columns) - set(df_test.columns)



def remove_outliers_iqr(df):
    df_clean = df.copy()
    for col in df:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Keep only rows within bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Cleaned data
df_train_clean = remove_outliers_iqr(df_train)


selected_features = df_train_clean.columns

# Automatically set rows and columns based on number of features
ncols = 2
nrows = math.ceil(len(selected_features) / ncols)

# Bigger figure size for better readability
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 3))
fig.suptitle('Boxplots of Selected Features', fontsize=20, weight='bold')
axes = axes.flatten()

# Plot boxplots for each feature
for i, col in enumerate(selected_features):
    sns.boxplot(y=df_train_clean[col], ax=axes[i], color='skyblue')
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove unused subplots if any
for j in range(len(selected_features), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout for neat spacing
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()



selected_features = df_train_clean.columns

# Automatically set rows and columns based on number of features
ncols = 2
nrows = math.ceil(len(selected_features) / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 3))
fig.suptitle('Histplot of Selected Features', fontsize=20, weight='bold')
axes = axes.flatten()

# Plot boxplots for each feature
for i, col in enumerate(selected_features):
    sns.histplot(x=df_train_clean[col], kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove unused subplots if any
for j in range(len(selected_features), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout for neat spacing
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(df_train_clean.corr(), annot=True, cmap='coolwarm',fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


sns.pairplot(df_train_clean, hue='servergetpoint', diag_kind='kde', plot_kws={'alpha':0.6})
plt.show()


X = df_train_clean.drop('servergetpoint', axis=1)
y = df_train_clean['servergetpoint']


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns


numeric_transformer = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='rmse'
    ))
])


model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# --- Train and predict ---
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluation metrics ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")


