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


try:
    import xgboost
except Exception:
    !pip install xgboost


import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


playground_data_path = kagglehub.competition_download('playground-series-s5e8')
train_data_path = os.path.join(playground_data_path, "train.csv")
test_data_path = os.path.join(playground_data_path, "test.csv")
sample_submission_path = os.path.join(playground_data_path, "sample_submission.csv")


le = LabelEncoder()


df_train = pd.read_csv(train_data_path).drop(['id'], axis=1)
df_test = pd.read_csv(test_data_path).drop(['id'], axis=1)
df_sample = pd.read_csv(sample_submission_path)


numerical_cols = df_train.select_dtypes(include=np.number).columns.to_list()
categorical_cols = df_train.select_dtypes(exclude=np.number).columns.to_list()


def preprocess_data(df):
    df = df.drop(['index', 'id'], axis=1, errors='ignore')
    is_train = 'y' in df.columns.to_list()

    if is_train:
        df['day_cat'] = le.fit_transform(df['day'].astype(str))
    else:
        df['day_cat'] = le.transform(df['day'].astype(str))

    df['contact_flag'] = (df['pdays'] != -1)

    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Transform skewed numerical features with clipping to avoid inf
    df['balance_log'] = np.log1p(np.clip(df['balance'], 0, None))  # Clip negative to 0
    df['duration_log'] = np.log1p(df['duration'])
    
    # Handle pdays=-1 with a flag
    df['pdays_flag'] = (df['pdays'] != -1).astype(int)


    return df
    
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)



df_train.head()


df_train.describe()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO  # Optional, if loading from string content

df = df_train.copy()

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
fig.suptitle('Distribution Histograms for Numerical Features', fontsize=16)
axs = axs.flatten()  # Flatten for easy indexing

for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], ax=axs[i], kde=True, bins=30)
    axs[i].set_title(f'Distribution of {col}')
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('Frequency')

# Hide unused subplots if any
for j in range(len(numerical_cols), len(axs)):
    axs[j].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot bar plots for categorical features in a grid (as "histograms" via counts)
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 20))
fig.suptitle('Distribution Bar Plots for Categorical Features', fontsize=16)
axs = axs.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(y=df[col], ax=axs[i], order=df[col].value_counts().index)
    axs[i].set_title(f'Distribution of {col}')
    axs[i].set_xlabel('Count')
    axs[i].set_ylabel(col)

# Hide unused subplots if any
for j in range(len(categorical_cols), len(axs)):
    axs[j].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


import scipy.stats as stats

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
fig.suptitle('Q-Q Plots for Numerical Features', fontsize=14)
axs = axs.flatten()  # Flatten for easy indexing

for i, col in enumerate(numerical_cols):
    # Generate Q-Q plot
    stats.probplot(df[col].dropna(), dist="norm", plot=axs[i])
    axs[i].set_title(f'Q-Q Plot: {col}', fontsize=10)
    axs[i].set_xlabel('Theoretical Quantiles')
    axs[i].set_ylabel('Sample Quantiles')
    # Adjust font sizes for compactness
    axs[i].tick_params(axis='both', labelsize=8)
    axs[i].title.set_size(10)

# Hide unused subplots if any
for j in range(len(numerical_cols), len(axs)):
    axs[j].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_train.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()


X = df_train.drop('y', axis=1)
y = df_train['y']

X_test = df_test.copy()

# Handle class imbalance with scale_pos_weight
scale_pos_weight = (len(y) - y.sum()) / y.sum()  # Approx 8:1 ratio

# Train final model with best params and enable_categorical
best_params = {
    'max_depth': 10,
    'learning_rate': 0.0242998981703963,
    'n_estimators': 1000,
    'subsample': 0.8599760307484523,
    'colsample_bytree': 0.7517201994404935,
    'gamma': 2.440775889248838,
    'min_child_weight': 10
}

final_model = XGBClassifier(
    **best_params,
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    tree_method = "hist", 
    device = "cuda",
    enable_categorical=True,  # Fix for category dtype
    random_state=42
)

final_model.fit(X, y)

# Predict and submit
test_preds = final_model.predict_proba(X_test)[:, 1]
df_sample['y'] = test_preds
df_sample.to_csv('submission.csv', index=False)
print('submission saved!')

