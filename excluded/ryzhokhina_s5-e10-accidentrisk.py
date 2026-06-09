# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


train_path = "/kaggle/input/playground-series-s5e10/train.csv"
test_path = "/kaggle/input/playground-series-s5e10/test.csv"
submission_path = "/kaggle/input/playground-series-s5e10/sample_submission.csv"


train = pd.read_csv(train_path, index_col= 0)


print(train.shape)
train.head()


train.describe()


all_columns = train.columns


all_columns


display(train.dtypes)


target_col = "accident_risk"


num_cols = train.select_dtypes(include = np.number).columns.to_list()
if target_col in num_cols:
    num_cols.remove(target_col)
print(num_cols)


cat_cols = train.select_dtypes(exclude = np.number).columns.to_list()
print(cat_cols)


print('\nMissing values per column:')
display(train.isna().sum().sort_values(ascending=False))


fig = plt.figure(figsize = (12,7))
sns.histplot(train[target_col], bins = 40,kde = True)
plt.title(f"Distribution of {target_col}")
plt.xlabel(target_col)
plt.ylabel('count')
plt.show()


print('Target summary:')
display(train[target_col].describe())


print(num_cols)


train[num_cols].hist(figsize=(12,10), bins=20)
plt.suptitle('Numerical Feature Distributions', y=0.93)
plt.show()


plt.figure(figsize=(6,4))
order = train['num_lanes'].value_counts().index
ax = sns.countplot(data=train, x='num_lanes', order=order)
ax.bar_label(ax.containers[0], fmt='%d', label_type='edge')
plt.title('NUM_LANES distribution')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
order = train['speed_limit'].value_counts().index
ax=sns.countplot(data=train, x='speed_limit', order=order)
ax.bar_label(ax.containers[0], fmt='%d', label_type='edge')
plt.title('speed_limit distribution')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
order = train['num_reported_accidents'].value_counts().index
ax=sns.countplot(data=train, x='num_reported_accidents', order=order)
ax.bar_label(ax.containers[0], fmt='%d', label_type='edge')
plt.title('num_reported_accidents distribution')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


for col in cat_cols:
        plt.figure(figsize=(6,4))
        order = train[col].value_counts().index
        ax = sns.countplot(data=train, x=col, order=order)
        ax.bar_label(ax.containers[0])
        plt.title(f'{col} distribution')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


for col in ['curvature', 'num_reported_accidents']:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=train, x=col, y=target_col, alpha=0.5)
    plt.title(f'{col} vs accident_risk')
    plt.tight_layout()
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=train, x=col, y=target_col)
    plt.title(f'Accident_risk by {col}')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def cat_risk_range_summary(frame, target='accident_risk', cats=None):
    if cats is None:
        cats = frame.select_dtypes(exclude=np.number).columns.tolist()
    rows = []
    for col in cats:
        means = frame.groupby(col)[target].mean().sort_values()
        rows.append({
            'feature': col,
            'min_mean': means.iloc[0],
            'max_mean': means.iloc[-1],
            'risk_range': means.iloc[-1] - means.iloc[0],
            'n_levels': means.shape[0]
        })
    return pd.DataFrame(rows).sort_values('risk_range', ascending=False)

cols = cat_cols + ['num_lanes', 'speed_limit', 'num_reported_accidents']
cat_impact = cat_risk_range_summary(train, target=target_col, cats=cols)
display(cat_impact)



from scipy.stats import f_oneway

rows = []
for col in cols:
    groups = [train.loc[train[col]==level, 'accident_risk'] for level in train[col].unique()]
    stat, p = f_oneway(*groups)
    rows.append({
            'feature': col,
            'p-value': p,
            'significant': p < 0.05,
        })
fr = pd.DataFrame(rows).sort_values('p-value', ascending=True)
display(fr)


print('Duplicate rows:', train.duplicated().sum())
for c in cat_cols:
    rare = (train[c].value_counts(normalize=True) < 0.01).sum()
    if rare:
        print(f"Warning: {c} has {rare} rare levels (<1% of rows)")

zero_var = [c for c in train.columns if train[c].nunique(dropna=False) == 1]
if zero_var:
    print('Zero-variance columns:', zero_var)
else:
    print('No zero-variance columns detected.')


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool


target = 'accident_risk'

categorical_features = [
    'road_type', 'lighting', 'weather', 'time_of_day'
]

# Ordinal / numeric features (already numeric)
numeric_features = [
    'speed_limit', 'num_lanes', 'curvature', 'num_reported_accidents'
]

# Binary features (already 0/1, keep numeric)
binary_features = [
    'public_road', 'holiday'
]



train[binary_features] = train[binary_features].astype(int)
print(train[binary_features].dtypes)


# --- Combine all features ---
features = categorical_features + numeric_features + binary_features

# --- 2ï¸�âƒ£ Split data ---
X = train[features].copy()
y = train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3ï¸�âƒ£ Prepare CatBoost pools ---
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
val_pool = Pool(X_val, y_val, cat_features=categorical_features)

# --- 4ï¸�âƒ£ Initialize model ---
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

# --- 5ï¸�âƒ£ Train ---
model.fit(train_pool, eval_set=val_pool, use_best_model=True)



# --- 6ï¸�âƒ£ Evaluate ---
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("\nğŸ“Š Baseline Model Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"RÂ² Score: {r2:.4f}")

# --- 7ï¸�âƒ£ Feature importance ---
importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.get_feature_importance(train_pool)
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(9,6))
sns.barplot(data=importances.head(12), x='Importance', y='Feature', palette='viridis')
plt.title('Top Feature Importances â€” CatBoost Baseline')
plt.tight_layout()
plt.show()

# --- 8ï¸�âƒ£ Residual analysis ---
residuals = y_val - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True, color='teal')
plt.title("Residual Distribution (y_true - y_pred)")
plt.xlabel("Residual")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.5)
plt.plot([0,1],[0,1],'r--')
plt.xlabel("True accident_risk")
plt.ylabel("Predicted accident_risk")
plt.title("Predicted vs Actual Accident Risk")
plt.show()


print(test_path)


test = pd.read_csv(test_path, index_col = 0)
print(test.shape)
test.head()


test[binary_features] = test[binary_features].astype(int)
print(test[binary_features].dtypes)


x_test = test[features].copy()


test_predict = model.predict(x_test)


len(test_predict)


submission = pd.read_csv(submission_path)


submission.head()


submission['accident_risk'] = test_predict


submission.head()


submission.to_csv('submission.csv', index=False)

