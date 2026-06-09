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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
df = df.drop_duplicates()


print(df.describe())


df.info()


df.nunique()


fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(8,20))
for i, col in enumerate(df.select_dtypes(include=[np.number]).columns):
    sns.histplot(df[col], bins=30, kde=True, ax=axs[i])
    axs[i].set_title(f'Histogram of {col}')
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12,10))

for i, col in enumerate(df.select_dtypes(include=[np.number]).drop('Calories', axis=1).columns):
    row = i // 2
    col_idx = i % 2
    sns.violinplot(x=df['Sex'], y=df[col], ax=axs[row, col_idx])
    axs[row, col_idx].set_title(f'Sex vs {col}')

plt.tight_layout()
plt.show()


sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


sns.pairplot(df)
plt.suptitle("Pairplot of All Features", y=1.02)
plt.show()


def feature_engineering(data, numeric_cols):
    data['BMI'] = data['Weight'] / (data['Height'] / 100) ** 2
    data['Intensity'] = data['Heart_Rate'] / data['Duration']
    
    for i in range(len(numeric_cols)):
        f1 = numeric_cols[i]
        for j in range(i+1, len(numeric_cols)):
            f2 = numeric_cols[j]
            data[f'{f1}_x_{f2}'] = data[f1] * data[f2]
    return data

numeric_cols = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']

label_enc = LabelEncoder()
df['Sex'] = label_enc.fit_transform(df['Sex'])
df_test['Sex'] = label_enc.transform(df_test['Sex'])

df = feature_engineering(df, numeric_cols)
df_test = feature_engineering(df_test, numeric_cols)

X = df.drop(['Calories'], axis=1)
y = np.log1p(df['Calories'].values)


pca = PCA(n_components=3)
X_pca = pca.fit_transform(X.drop('Sex', axis=1))

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_pca) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.plot(range(1,10), inertias, "bo-")
plt.xlabel("$k$")
plt.ylabel("Inertia")
plt.grid(True)
plt.title("Elbow Method for Optimal K")
plt.show()

# Using 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_pca)
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
pca_df['cluster'] = kmeans.labels_
pca_df['target_group'] = pd.qcut(df['Calories'], q=3, labels=False)

# Comparing clusters with groups
accuracy = (pca_df['cluster'] == pca_df['target_group']).mean()
print(f"Accuracy between clusters and target groups: {accuracy:.2%}")

# visualization of clusters
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'], c=pca_df['target_group'], cmap='viridis', alpha=0.7)
plt.title("3D PCA Projection with Target Group Coloring")
plt.colorbar(scatter, label='Target Group')
plt.show()


import lightgbm as lgbm
FOLDS = 2
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

cat_preds = np.zeros((FOLDS, len(df_test)))
xgb_preds = np.zeros((FOLDS, len(df_test)))
lgbm_preds = np.zeros((FOLDS, len(df_test)))

oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_lgbm = np.zeros(len(X))

models = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#' * 15} Fold {fold+1} {'#' * 15}")
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

    # CATBOOST
    cat_model = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.02,
        depth=12,
        l2_leaf_reg=3,
        verbose=1000,
        early_stopping_rounds=200,
        loss_function='RMSE',
        eval_metric='RMSE',
        task_type='CPU'
    )
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True, cat_features=[0])

    # XGBOOST
    xgb_model = XGBRegressor(
        max_depth=10,
        n_estimators=2000,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.75,
        gamma=0.01,
        enable_categorical=True,
        tree_method="hist",
        device="cpu"
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, verbose=0)

   # LIGHTGBM
    lgbm_model = LGBMRegressor(
        num_leaves=50,
        max_depth=10,
        learning_rate=0.01,
        n_estimators=3000,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=1,
        reg_lambda=1,
        verbose=100,  
    )

    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgbm.early_stopping(100)],
    )

    # OOF Predictions
    oof_cat[valid_idx] = cat_model.predict(X_valid)
    oof_xgb[valid_idx] = xgb_model.predict(X_valid)
    oof_lgbm[valid_idx] = lgbm_model.predict(X_valid)

    # Test predictions
    cat_preds[fold] = cat_model.predict(df_test)
    xgb_preds[fold] = xgb_model.predict(df_test)
    lgbm_preds[fold] = lgbm_model.predict(df_test)

    # RMSE
    rmse_cat = mean_squared_error(y_valid, oof_cat[valid_idx]) ** 0.5
    rmse_xgb = mean_squared_error(y_valid, oof_xgb[valid_idx]) ** 0.5
    rmse_lgbm = mean_squared_error(y_valid, oof_lgbm[valid_idx]) ** 0.5
    print(f'CAT_RMSE: {rmse_cat:.4f}, XGB_RMSE: {rmse_xgb:.4f}, LGBM_RMSE: {rmse_lgbm:.4f}')

# Average for all folds
pred_cat = np.expm1(np.mean(cat_preds, axis=0))
pred_xgb = np.expm1(np.mean(xgb_preds, axis=0))
pred_lgbm = np.expm1(np.mean(lgbm_preds, axis=0))

# Ensemble weighted average
final_pred = pred_cat * 0.3 + pred_xgb * 0.3 + pred_lgbm * 0.4
final_pred = np.clip(final_pred, 1, 314)


submission = pd.DataFrame({
    'id': df_test.index,
    'Calories': final_pred 
})

# Сохранение в CSV
submission.to_csv('submission.csv', index=False)
print("Файл submission.csv успешно сохранен!")
submission.head()


plt.figure(figsize=(10,6))
sns.kdeplot(np.expm1(oof_cat), label='CatBoost OOF', fill=True, alpha=0.3)
sns.kdeplot(np.expm1(oof_xgb), label='XGBoost OOF', fill=True, alpha=0.3)
sns.kdeplot(np.expm1(oof_lgbm), label='LightGBM OOF', fill=True, alpha=0.3)
sns.kdeplot(df['Calories'], label='True Calories', fill=True, alpha=0.3)
plt.legend()
plt.title("OOF Prediction vs True Values")
plt.xlabel("Calories")
plt.ylabel("Density")
plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    df['Age'],
    df['Weight'],
    df['Heart_Rate'],
    c=np.expm1(oof_cat),
    cmap='viridis',
    s=50,
    alpha=0.8
)
plt.colorbar(sc, ax=ax, label='Predicted Calories')
ax.set_xlabel("Age")
ax.set_ylabel("Weight")
ax.set_zlabel("Heart Rate")
plt.title("3D Plot of Predicted Calories (CatBoost)")
plt.show()


print("\nFinal Model Scores:")
print(f"CatBoost RMSE: {mean_squared_error(y, oof_cat) ** 0.5:.4f}")
print(f"XGBoost RMSE: {mean_squared_error(y, oof_xgb) ** 0.5:.4f}")
print(f"LightGBM RMSE: {mean_squared_error(y, oof_lgbm) ** 0.5:.4f}")

