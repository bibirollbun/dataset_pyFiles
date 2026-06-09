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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  
import plotly.io as pio 
import optuna
import time

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff  
pio.renderers.default = 'iframe_connected' 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool



import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission_data = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train_data.head(5)


train_data.info()


train_data.describe()


train_data.isnull().sum()


object_column_names = train_data.select_dtypes(include=['object']).columns
print("Object Column Names:", object_column_names.tolist())


numerical_column_names = train_data.select_dtypes(include=['number']).columns
print("Numerical Column Names:", numerical_column_names.tolist())


color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']



plt.figure(figsize=(10, 5))
sns.histplot(train_data['Calories'], bins=50, kde=True, color='#4dd0e1', edgecolor='black')
plt.title('Distribution of Calories Burned', fontsize=16, weight='bold')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


sex_count = train_data['Sex'].value_counts().reset_index()
sex_count.columns = ['Sex','Count']

fig = px.pie(
    sex_count,
    names ='Sex',
    values = "Count",
    color = 'Sex',
    color_discrete_sequence = color_palette,
    title = "Sex Distribution"
)
fig.update_traces(textinfo = 'percent+label')
fig.update_layout(width = 500, height=500)

fig.show()


fig, axes = plt.subplots(nrows=len(cols), ncols=1, figsize=(8, 18))

for i, col in enumerate(cols):
    sns.histplot(train_data[col], bins=50, kde=True, ax=axes[i], color=color_palette[i % len(color_palette)], edgecolor='black', linewidth=0.5)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


fig = px.scatter(
    train_data,
    x='Heart_Rate',
    y='Calories',
    color='Sex',
    title='Calories Burned vs Heart Rate',
    labels={
        'Heart_Rate': 'Heart Rate (bpm)',
        'Calories': 'Calories Burned (kcal)',
        'Sex': 'Gender'
    },
     color_discrete_sequence=color_palette
)
fig.show()


fig = px.scatter(
    train_data,
    x='Height',
    y='Weight',
    color='Sex',
    title='Height & Weight vs Heart Rate',
    labels={
        'Height': 'Height',
        'Weight': 'Weight',
        'Sex': 'Gender'
    },
    color_discrete_sequence=color_palette
)

fig.show()


sorted_data = train_data.sort_values('Duration')
fig = px.line(
    sorted_data,
    x='Duration',
    y='Calories',
    markers=True,
    title='Calories Burned vs Exercise Duration',
    color_discrete_sequence=color_palette,
    labels={'Duration': 'Exercise Duration (minutes)', 'Calories': 'Calories Burned'}
)

fig.update_layout(
    width=700,
    height=400,
    font=dict(size=14),
    plot_bgcolor='white',
    title_x=0.5
)

fig.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Violin Plot: Body_Temp vs Sex
sns.violinplot(x='Sex', y='Body_Temp', data=train_data, ax=axes[0], palette=color_palette)
axes[0].set_title('Sex vs Body Temp')
axes[0].set_xlabel('Sex')
axes[0].set_ylabel('Body Temperature')

# 2. Violin Plot: Heart_Rate vs Sex
sns.violinplot(x='Sex', y='Heart_Rate', data=train_data, ax=axes[1], palette=color_palette)
axes[1].set_title('Sex vs Heart Rate')
axes[1].set_xlabel('Sex')
axes[1].set_ylabel('Heart Rate')

# 3. Violin Plot: Calories vs Sex
sns.violinplot(x='Sex', y='Calories', data=train_data, ax=axes[2], palette=color_palette)
axes[2].set_title('Sex vs Calories')
axes[2].set_xlabel('Sex')
axes[2].set_ylabel('Calories Burned')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


sns.kdeplot(train_data, x="Calories", hue="Sex", fill=True)


for col in cols:
    print(f"\nAverage {col} by Sex:")
    print(train_data.groupby('Sex')[col].mean().round(2))
    print("-"*30)


corr = train_data[cols].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


le =  LabelEncoder()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
train_data['BMI'] = train_data['Weight'] / ( (train_data['Height'] / 100) ** 2 )
train_data['HR_per_min'] = train_data['Heart_Rate'] / train_data['Duration']
train_data['Temp_diff_from_norm'] = train_data['Body_Temp'] - 37.0


test_data['Sex'] = le.fit_transform(test_data['Sex'])
test_data['BMI'] = test_data['Weight'] / ( (test_data['Height'] / 100) ** 2 )
test_data['HR_per_min'] = test_data['Heart_Rate'] / test_data['Duration']
test_data['Temp_diff_from_norm'] = test_data['Body_Temp'] - 37.0


train_data.head()


drop_col = 'Calories'
X = train_data.drop(columns=[drop_col])
y = np.log1p(train_data[drop_col].values)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


train_pool = Pool(X_train, y_train)
valid_pool = Pool(X_valid, y_valid)

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 5000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        'depth': trial.suggest_int('depth', 4, 12),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-2, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'RMSE',
        'eval_metric': 'MSLE',
        'random_seed': 42,
        'early_stopping_rounds': 50,
        'verbose': False,
        'task_type': 'GPU'  # Use GPU for training
    }

    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True
    )
    
    # Predict on validation set
    preds = model.predict(X_valid)
    # Compute RMSLE
    rmsle = mean_squared_log_error(y_valid, preds, squared=False)
    return rmsle


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Out-of-fold and test predictions
oof_preds = np.zeros(len(train_data))
test_preds_log = np.zeros(len(test_data))

# Best CatBoost parameters from Optuna
params = {
    'iterations': 3458,
    'learning_rate': 0.0666715186029617,
    'depth': 10,
    'l2_leaf_reg': 2.03429411556143,
    'bagging_temperature': 0.08412755040615273,
    'border_count': 222,
    'loss_function': 'RMSE',
    'eval_metric': 'MSLE',
    'random_seed': 42,
    'verbose': False
}

# K-fold training loop
for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=50,
        use_best_model=True
    )

    # Out-of-fold predictions
    oof_preds[valid_idx] = model.predict(X_valid)

    # Test set predictions (log scale)
    test_preds_log += model.predict(test_data) / n_splits

    print(f"Fold {fold + 1} completed.")

# Convert log predictions back to original scale
oof_preds_exp = np.expm1(oof_preds)
test_preds = np.expm1(test_preds_log)

# Optionally evaluate OOF performance
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred))))

oof_score = rmsle(train_data[drop_col].values, oof_preds_exp)
print(f"OOF RMSLE: {oof_score:.5f}")


df_submission = submission_data.copy()
df_submission[drop_col] = test_preds

df_submission.to_csv('submission.csv', index=False)
print(df_submission.head())

